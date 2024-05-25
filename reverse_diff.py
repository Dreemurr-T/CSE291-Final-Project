import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff
import string
import random
import pretty_print

# From https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
def random_id_generator(size=6, chars=string.ascii_lowercase + string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def reverse_diff(diff_func_id : str,
                 structs : dict[str, loma_ir.Struct],
                 funcs : dict[str, loma_ir.func],
                 diff_structs : dict[str, loma_ir.Struct],
                 func : loma_ir.FunctionDef,
                 func_to_rev : dict[str, str]) -> loma_ir.FunctionDef:
    """ Given a primal loma function func, apply reverse differentiation
        and return a function that computes the total derivative of func.

        For example, given the following function:
        def square(x : In[float]) -> float:
            return x * x
        and let diff_func_id = 'd_square', reverse_diff() should return
        def d_square(x : In[float], _dx : Out[float], _dreturn : float):
            _dx = _dx + _dreturn * x + _dreturn * x

        Parameters:
        diff_func_id - the ID of the returned function
        structs - a dictionary that maps the ID of a Struct to 
                the corresponding Struct
        funcs - a dictionary that maps the ID of a function to 
                the corresponding func
        diff_structs - a dictionary that maps the ID of the primal
                Struct to the corresponding differential Struct
                e.g., diff_structs['float'] returns _dfloat
        func - the function to be differentiated
        func_to_rev - mapping from primal function ID to its reverse differentiation
    """

    # Some utility functions you can use for your homework.
    def type_to_string(t):
        match t:
            case loma_ir.Int():
                return 'int'
            case loma_ir.Float():
                return 'float'
            case loma_ir.Array():
                return 'array_' + type_to_string(t.t)
            case loma_ir.Struct():
                return t.id
            case _:
                assert False

    def var_to_differential(expr, var_to_dvar):
        match expr:
            case loma_ir.Var():
                return loma_ir.Var(var_to_dvar[expr.id], t = expr.t)
            case loma_ir.ArrayAccess():
                return loma_ir.ArrayAccess(\
                    var_to_differential(expr.array, var_to_dvar),
                    expr.index,
                    t = expr.t)
            case loma_ir.StructAccess():
                return loma_ir.StructAccess(\
                    var_to_differential(expr.struct, var_to_dvar),
                    expr.member_id,
                    t = expr.t)
            case _:
                assert False

    def assign_zero(target):
        match target.t:
            case loma_ir.Int():
                return []
            case loma_ir.Float():
                return [loma_ir.Assign(target, loma_ir.ConstFloat(0.0))]
            case loma_ir.Struct():
                s = target.t
                stmts = []
                for m in s.members:
                    target_m = loma_ir.StructAccess(
                        target, m.id, t = m.t)
                    if isinstance(m.t, loma_ir.Float):
                        stmts += assign_zero(target_m)
                    elif isinstance(m.t, loma_ir.Int):
                        pass
                    elif isinstance(m.t, loma_ir.Struct):
                        stmts += assign_zero(target_m)
                    else:
                        assert isinstance(m.t, loma_ir.Array)
                        assert m.t.static_size is not None
                        for i in range(m.t.static_size):
                            target_m = loma_ir.ArrayAccess(
                                target_m, loma_ir.ConstInt(i), t = m.t.t)
                            stmts += assign_zero(target_m)
                return stmts
            case _:
                assert False

    def accum_deriv(target, deriv, overwrite):
        match target.t:
            case loma_ir.Int():
                return []
            case loma_ir.Float():
                if overwrite:
                    return [loma_ir.Assign(target, deriv)]
                else:
                    #return [loma_ir.Assign(target,
                    #    loma_ir.BinaryOp(loma_ir.Add(), target, deriv))]
                    return [loma_ir.CallStmt(loma_ir.Call('atomic_add',
                        [target, deriv]))]
            case loma_ir.Struct():
                s = target.t
                stmts = []
                for m in s.members:
                    target_m = loma_ir.StructAccess(
                        target, m.id, t = m.t)
                    deriv_m = loma_ir.StructAccess(
                        deriv, m.id, t = m.t)
                    if isinstance(m.t, loma_ir.Float):
                        stmts += accum_deriv(target_m, deriv_m, overwrite)
                    elif isinstance(m.t, loma_ir.Int):
                        pass
                    elif isinstance(m.t, loma_ir.Struct):
                        stmts += accum_deriv(target_m, deriv_m, overwrite)
                    else:
                        assert isinstance(m.t, loma_ir.Array)
                        assert m.t.static_size is not None
                        for i in range(m.t.static_size):
                            target_m = loma_ir.ArrayAccess(
                                target_m, loma_ir.ConstInt(i), t = m.t.t)
                            deriv_m = loma_ir.ArrayAccess(
                                deriv_m, loma_ir.ConstInt(i), t = m.t.t)
                            stmts += accum_deriv(target_m, deriv_m, overwrite)
                return stmts
            case _:
                assert False

    def check_lhs_is_output_arg(lhs, output_args):
        match lhs:
            case loma_ir.Var():
                return lhs.id in output_args
            case loma_ir.StructAccess():
                return check_lhs_is_output_arg(lhs.struct, output_args)
            case loma_ir.ArrayAccess():
                return check_lhs_is_output_arg(lhs.array, output_args)
            case _:
                assert False

    # A utility class that you can use for HW3.
    # This mutator normalizes each call expression into
    # f(x0, x1, ...)
    # where x0, x1, ... are all loma_ir.Var or 
    # loma_ir.ArrayAccess or loma_ir.StructAccess
    # Furthermore, it normalizes all Assign statements
    # with a function call
    # z = f(...)
    # into a declaration followed by an assignment
    # _tmp : [z's type]
    # _tmp = f(...)
    # z = _tmp
    class CallNormalizeMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            self.tmp_count = 0
            self.tmp_declare_stmts = []
            new_body = [self.mutate_stmt(stmt) for stmt in node.body]
            new_body = irmutator.flatten(new_body)

            new_body = self.tmp_declare_stmts + new_body

            return loma_ir.FunctionDef(\
                node.id, node.args, new_body, node.is_simd, node.ret_type, lineno = node.lineno)

        def mutate_return(self, node):
            self.tmp_assign_stmts = []
            val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Return(\
                val,
                lineno = node.lineno)]

        def mutate_declare(self, node):
            self.tmp_assign_stmts = []
            val = None
            if node.val is not None:
                val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Declare(\
                node.target,
                node.t,
                val,
                lineno = node.lineno)]

        def mutate_assign(self, node):
            self.tmp_assign_stmts = []
            target = self.mutate_expr(node.target)
            self.has_call_expr = False
            val = self.mutate_expr(node.val)
            if self.has_call_expr:
                # turn the assignment into a declaration plus
                # an assignment
                self.tmp_count += 1
                tmp_name = f'_call_t_{self.tmp_count}_{random_id_generator()}'
                self.tmp_count += 1
                self.tmp_declare_stmts.append(loma_ir.Declare(\
                    tmp_name,
                    target.t,
                    lineno = node.lineno))
                tmp_var = loma_ir.Var(tmp_name, t = target.t)
                assign_tmp = loma_ir.Assign(\
                    tmp_var,
                    val,
                    lineno = node.lineno)
                assign_target = loma_ir.Assign(\
                    target,
                    tmp_var,
                    lineno = node.lineno)
                return self.tmp_assign_stmts + [assign_tmp, assign_target]
            else:
                return self.tmp_assign_stmts + [loma_ir.Assign(\
                    target,
                    val,
                    lineno = node.lineno)]

        def mutate_call_stmt(self, node):
            self.tmp_assign_stmts = []
            call = self.mutate_expr(node.call)
            return self.tmp_assign_stmts + [loma_ir.CallStmt(\
                call,
                lineno = node.lineno)]

        def mutate_call(self, node):
            self.has_call_expr = True
            new_args = []
            for arg in node.args:
                if not isinstance(arg, loma_ir.Var) and \
                        not isinstance(arg, loma_ir.ArrayAccess) and \
                        not isinstance(arg, loma_ir.StructAccess):
                    arg = self.mutate_expr(arg)
                    tmp_name = f'_call_t_{self.tmp_count}_{random_id_generator()}'
                    self.tmp_count += 1
                    tmp_var = loma_ir.Var(tmp_name, t = arg.t)
                    self.tmp_declare_stmts.append(loma_ir.Declare(\
                        tmp_name, arg.t))
                    self.tmp_assign_stmts.append(loma_ir.Assign(\
                        tmp_var, arg))
                    new_args.append(tmp_var)
                else:
                    new_args.append(arg)
            return loma_ir.Call(node.id, new_args, t = node.t)

    class ForwardPassMutator(irmutator.IRMutator):
        def __init__(self, output_args):
            self.output_args = output_args
            self.cache_vars_list = {}
            self.var_to_dvar = {}
            self.type_cache_size = {}
            self.type_to_stack_and_ptr_names = {}
            self.loop_vars_dict = {}
            self.loop_count = 0
            self.loop_var_declare_stmts = []
            self.parent_loop_size = []

        def mutate_return(self, node):
            return []

        def mutate_declare(self, node):
            # For each declaration, add another declaration for the derivatives
            # except when it's an integer
            if node.t != loma_ir.Int():
                dvar = '_d' + node.target + '_' + random_id_generator()
                self.var_to_dvar[node.target] = dvar
                return [node, loma_ir.Declare(\
                    dvar,
                    node.t,
                    lineno = node.lineno)]
            else:
                return node

        def mutate_assign(self, node):
            if check_lhs_is_output_arg(node.target, self.output_args):
                return []

            # y = f(x0, x1, ..., y)
            # we will use a temporary array _t to hold variable y for later use:
            # _t[stack_pos++] = y
            # y = f(x0, x1, ..., y)
            assign_primal = loma_ir.Assign(\
                node.target,
                self.mutate_expr(node.val),
                lineno = node.lineno)
            # backup
            t_str = type_to_string(node.val.t)
            if t_str in self.type_to_stack_and_ptr_names:
                stack_name, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
            else:
                random_id = random_id_generator()
                stack_name = f'_t_{t_str}_{random_id}'
                stack_ptr_name = f'_stack_ptr_{t_str}_{random_id}'
                self.type_to_stack_and_ptr_names[t_str] = (stack_name, stack_ptr_name)
            
            stack_ptr_var = loma_ir.Var(stack_ptr_name, t=loma_ir.Int())
            cache_var_expr = loma_ir.ArrayAccess(
                loma_ir.Var(stack_name),
                stack_ptr_var,
                t = node.val.t)
            cache_primal = loma_ir.Assign(cache_var_expr, node.target)
            stack_advance = loma_ir.Assign(stack_ptr_var,
                loma_ir.BinaryOp(loma_ir.Add(), stack_ptr_var, loma_ir.ConstInt(1)))

            if node.val.t in self.cache_vars_list:
                self.cache_vars_list[node.val.t].append((cache_var_expr, node.target))
            else:
                self.cache_vars_list[node.val.t] = [(cache_var_expr, node.target)]
            if node.val.t in self.type_cache_size:
                self.type_cache_size[node.val.t] += 1
            else:
                self.type_cache_size[node.val.t] = 1
            return [cache_primal, stack_advance, assign_primal]

        def mutate_call_stmt(self, node):
            call_expr = node.call
            for arg_expr in call_expr.args:
                if check_lhs_is_output_arg(arg_expr, self.output_args):
                    return []

            # similar to assign: backup all outputs of the function
            stmts = []
            if call_expr.id != 'atomic_add':
                args = funcs[call_expr.id].args
            else:
                args = [loma_ir.Arg('target', loma_ir.Float(), loma_ir.Out()),
                        loma_ir.Arg('source', loma_ir.Float(), loma_ir.In())]
            for i, f_arg in enumerate(args):
                if f_arg.i == loma_ir.Out():
                    arg_expr = call_expr.args[i]
                    t_str = type_to_string(f_arg.t)
                    if t_str in self.type_to_stack_and_ptr_names:
                        stack_name, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
                    else:
                        random_id = random_id_generator()
                        stack_name = f'_t_{t_str}_{random_id}'
                        stack_ptr_name = f'_stack_ptr_{t_str}_{random_id}'
                        self.type_to_stack_and_ptr_names[t_str] = (stack_name, stack_ptr_name)
                    stack_ptr_var = loma_ir.Var(stack_ptr_name, t=loma_ir.Int())
                    cache_var_expr = loma_ir.ArrayAccess(
                        loma_ir.Var(stack_name),
                        stack_ptr_var,
                        t = f_arg.t)
                    stmts.append(loma_ir.Assign(cache_var_expr, arg_expr))
                    stmts.append(loma_ir.Assign(stack_ptr_var,
                        loma_ir.BinaryOp(loma_ir.Add(), stack_ptr_var, loma_ir.ConstInt(1))))

                    if f_arg.t in self.cache_vars_list:
                        self.cache_vars_list[f_arg.t].append((cache_var_expr, arg_expr))
                    else:
                        self.cache_vars_list[f_arg.t] = [(cache_var_expr, arg_expr)]
                    if f_arg.t in self.type_cache_size:
                        self.type_cache_size[f_arg.t] += 1
                    else:
                        self.type_cache_size[f_arg.t] = 1
            stmts.append(node)
            return stmts

        def mutate_while(self, node):
            stmts = []

            # add a loop variable
            loop_var_name = f'_loop_var_{self.loop_count}_{random_id_generator()}'
            self.loop_count += 1
            loop_var = loma_ir.Var(loop_var_name, t=loma_ir.Int())
            stmts.append(loma_ir.Assign(\
                loop_var, loma_ir.ConstInt(0)))
            needs_pop_from_stack = len(self.parent_loop_size) > 0
            self.loop_vars_dict[node] = (loop_var, needs_pop_from_stack)
            self.loop_var_declare_stmts.append(\
                loma_ir.Declare(loop_var_name, loma_ir.Int()))

            # backup type_cache_size
            old_type_cache_size = self.type_cache_size
            self.type_cache_size = {}
            prev_loop_size = self.parent_loop_size[-1] if len(self.parent_loop_size) > 0 else 1
            self.parent_loop_size.append(prev_loop_size * node.max_iter)
            new_body = [self.mutate_stmt(stmt) for stmt in node.body]
            self.parent_loop_size.pop()
            new_body = irmutator.flatten(new_body)

            # merge old_type_cache_size and self.type_cache_size
            for t, size in self.type_cache_size.items():
                size = size * node.max_iter
                # add to old_type_cache_size
                if t in old_type_cache_size:
                    old_type_cache_size[t] += size
                else:
                    old_type_cache_size[t] = size
            self.type_cache_size = old_type_cache_size

            # increment of the loop variable
            new_body.append(loma_ir.Assign(\
                loop_var, 
                loma_ir.BinaryOp(loma_ir.Add(), loop_var, loma_ir.ConstInt(1))))

            stmts.append(loma_ir.While(node.cond, node.max_iter, new_body))

            # if the loop is inside another loop, then push the loop variable to
            # a stack 
            if len(self.parent_loop_size) > 0:
                loop_var_stack_name = loop_var_name + '_stack'
                loop_var_stack_ptr_name = loop_var_name + '_stack_ptr'
                loop_var_stack_ptr = loma_ir.Var(loop_var_stack_ptr_name, t = loma_ir.Int())
                loop_var_stack = loma_ir.ArrayAccess(\
                    loma_ir.Var(loop_var_stack_name),
                    loop_var_stack_ptr)
                stmts.append(loma_ir.Assign(loop_var_stack, loop_var))
                stmts.append(loma_ir.Assign(loop_var_stack_ptr,
                    loma_ir.BinaryOp(loma_ir.Add(), loop_var_stack_ptr, loma_ir.ConstInt(1))))

                self.loop_var_declare_stmts.append(loma_ir.Declare(\
                    loop_var_stack_name, loma_ir.Array(loma_ir.Int(), self.parent_loop_size[-1])))
                self.loop_var_declare_stmts.append(loma_ir.Declare(\
                    loop_var_stack_ptr_name, loma_ir.Int()))

            return stmts

    # HW2 happens here. Modify the following IR mutators to perform
    # reverse differentiation.
    class RevDiffMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            cnm = CallNormalizeMutator()
            node = cnm.mutate_function(node)

            random.seed(hash(node.id))
            # Each input argument is followed by an output (the adjoint)
            # Each output is turned into an input
            # The return value turn into an input
            self.var_to_dvar = {}
            new_args = []
            self.output_args = set()
            for arg in node.args:
                if arg.i == loma_ir.In():
                    new_args.append(arg)
                    dvar_id = '_d' + arg.id + '_' + random_id_generator()
                    new_args.append(loma_ir.Arg(dvar_id, arg.t, i = loma_ir.Out()))
                    self.var_to_dvar[arg.id] = dvar_id
                else:
                    assert arg.i == loma_ir.Out()
                    self.output_args.add(arg.id)
                    new_args.append(loma_ir.Arg(arg.id, arg.t, i = loma_ir.In()))
                    self.var_to_dvar[arg.id] = arg.id
            if node.ret_type is not None:
                self.return_var_id = '_dreturn_' + random_id_generator()
                new_args.append(loma_ir.Arg(self.return_var_id, node.ret_type, i = loma_ir.In()))

            # Forward pass
            fm = ForwardPassMutator(self.output_args)
            forward_body = node.body
            mutated_forward = [fm.mutate_stmt(fwd_stmt) for fwd_stmt in forward_body]
            mutated_forward = irmutator.flatten(mutated_forward)
            mutated_forward = fm.loop_var_declare_stmts + mutated_forward
            self.var_to_dvar = self.var_to_dvar | fm.var_to_dvar
            self.loop_vars_dict = fm.loop_vars_dict

            self.cache_vars_list = fm.cache_vars_list
            self.type_cache_size = fm.type_cache_size

            self.cache_vars_list = fm.cache_vars_list
            self.type_cache_size = fm.type_cache_size
            self.type_to_stack_and_ptr_names = fm.type_to_stack_and_ptr_names

            tmp_declares = []
            for t, exprs in fm.cache_vars_list.items():
                t_str = type_to_string(t)
                stack_name, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
                tmp_declares.append(loma_ir.Declare(stack_name,
                    loma_ir.Array(t, self.type_cache_size[t])))
                tmp_declares.append(loma_ir.Declare(stack_ptr_name,
                    loma_ir.Int(), loma_ir.ConstInt(0)))
            mutated_forward = tmp_declares + mutated_forward

            # Reverse pass
            self.adj_count = 0
            self.in_assign = False
            self.adj_declaration = []
            reversed_body = [self.mutate_stmt(stmt) for stmt in reversed(node.body)]
            reversed_body = irmutator.flatten(reversed_body)

            return loma_ir.FunctionDef(\
                diff_func_id,
                new_args,
                mutated_forward + self.adj_declaration + reversed_body,
                node.is_simd,
                ret_type = None,
                lineno = node.lineno)

        def mutate_return(self, node):
            # Propagate to each variable used in node.val
            self.adj = loma_ir.Var(self.return_var_id, t = node.val.t)
            return self.mutate_expr(node.val)

        def mutate_declare(self, node):
            if node.val is not None:
                if node.t == loma_ir.Int():
                    return []

                self.adj = loma_ir.Var(self.var_to_dvar[node.target])
                return self.mutate_expr(node.val)
            else:
                return []

        def mutate_assign(self, node):
            if node.val.t == loma_ir.Int():
                stmts = []
                # restore the previous value of this assignment
                t_str = type_to_string(node.val.t)
                _, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
                stack_ptr_var = loma_ir.Var(stack_ptr_name, t=loma_ir.Int())
                stmts.append(loma_ir.Assign(stack_ptr_var,
                    loma_ir.BinaryOp(loma_ir.Sub(), stack_ptr_var, loma_ir.ConstInt(1))))
                cache_var_expr, cache_target = self.cache_vars_list[node.val.t].pop()
                stmts.append(loma_ir.Assign(cache_target, cache_var_expr))
                return stmts

            self.adj = var_to_differential(node.target, self.var_to_dvar)
            if check_lhs_is_output_arg(node.target, self.output_args):
                # if the lhs is an output argument, then we can safely
                # treat this statement the same as "declare"
                return self.mutate_expr(node.val)
            else:
                stmts = []
                # restore the previous value of this assignment
                t_str = type_to_string(node.val.t)
                _, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
                stack_ptr_var = loma_ir.Var(stack_ptr_name, t=loma_ir.Int())
                stmts.append(loma_ir.Assign(stack_ptr_var,
                    loma_ir.BinaryOp(loma_ir.Sub(), stack_ptr_var, loma_ir.ConstInt(1))))
                cache_var_expr, cache_target = self.cache_vars_list[node.val.t].pop()
                stmts.append(loma_ir.Assign(cache_target, cache_var_expr))
                
                # First pass: accumulate
                self.in_assign = True
                self.adj_accum_stmts = []
                stmts += self.mutate_expr(node.val)
                self.in_assign = False

                # zero the target differential
                stmts += assign_zero(var_to_differential(node.target, self.var_to_dvar))

                # Accumulate the adjoints back to the target locations
                stmts += self.adj_accum_stmts
                return stmts

        def mutate_ifelse(self, node):
            # run the then statements and else statements backwards
            reversed_else = [self.mutate_stmt(stmt) for stmt in reversed(node.else_stmts)]
            reversed_else = irmutator.flatten(reversed_else)
            reversed_then = [self.mutate_stmt(stmt) for stmt in reversed(node.then_stmts)]
            reversed_then = irmutator.flatten(reversed_then)
            return loma_ir.IfElse(node.cond, reversed_then, reversed_else,
                lineno = node.lineno)

        def mutate_call_stmt(self, node):
            self.adj = loma_ir.ConstFloat(0.0)
            call_expr = node.call
            if call_expr.id != 'atomic_add':
                args = funcs[call_expr.id].args
            else:
                args = [loma_ir.Arg('target', loma_ir.Float(), loma_ir.Out()),
                        loma_ir.Arg('source', loma_ir.Float(), loma_ir.In())]

            stmts = []
            needs_restore = False
            for i, f_arg in enumerate(args):
                if f_arg.i == loma_ir.Out():
                    if not check_lhs_is_output_arg(call_expr.args[i], self.output_args):
                        needs_restore = True
            if needs_restore:
                # restore the previous values of the output variables
                for f_arg in reversed(args):
                    if f_arg.i == loma_ir.Out():
                        t_str = type_to_string(f_arg.t)
                        _, stack_ptr_name = self.type_to_stack_and_ptr_names[t_str]
                        stack_ptr_var = loma_ir.Var(stack_ptr_name, t=loma_ir.Int())
                        stmts.append(loma_ir.Assign(stack_ptr_var,
                            loma_ir.BinaryOp(loma_ir.Sub(), stack_ptr_var, loma_ir.ConstInt(1))))
                        cache_var_expr, cache_target = self.cache_vars_list[f_arg.t].pop()
                        stmts.append(loma_ir.Assign(cache_target, cache_var_expr))

            # Accumulate derivatives
            stmts += self.mutate_expr(node.call)
            
            if needs_restore:
                # zero the output differentials
                for i, f_arg in enumerate(args):
                    arg_expr = call_expr.args[i]
                    if f_arg.i == loma_ir.Out():
                        stmts += assign_zero(var_to_differential(arg_expr, self.var_to_dvar))

            return stmts

        def mutate_while(self, node):
            loop_var, needs_pop_from_stack = self.loop_vars_dict[node]
            stmts = []
            if needs_pop_from_stack:
                # loop_var = loop_var_stack.pop()
                assert isinstance(loop_var, loma_ir.Var)
                loop_var_stack_name = loop_var.id + '_stack'
                loop_var_stack_ptr_name = loop_var.id + '_stack_ptr'
                loop_var_stack_ptr = loma_ir.Var(loop_var_stack_ptr_name, t = loma_ir.Int())
                stmts.append(loma_ir.Assign(loop_var_stack_ptr,
                    loma_ir.BinaryOp(loma_ir.Sub(), loop_var_stack_ptr, loma_ir.ConstInt(1))))
                stmts.append(loma_ir.Assign(loop_var,
                    loma_ir.ArrayAccess(loma_ir.Var(loop_var_stack_name), loop_var_stack_ptr)))
            cond = loma_ir.BinaryOp(loma_ir.Greater(),
                loop_var, loma_ir.ConstInt(0))
            # run the body backwards
            reversed_body = [self.mutate_stmt(stmt) for stmt in reversed(node.body)]
            reversed_body = irmutator.flatten(reversed_body)
            # loop_var -= 1
            reversed_body.append(loma_ir.Assign(\
                loop_var, 
                loma_ir.BinaryOp(loma_ir.Sub(), loop_var, loma_ir.ConstInt(1))))
            stmts += [loma_ir.While(cond, node.max_iter, reversed_body)]
            return stmts

        def mutate_var(self, node):
            if self.in_assign:
                target = f'_adj_{str(self.adj_count)}'
                self.adj_count += 1
                self.adj_declaration.append(loma_ir.Declare(target, t=node.t))
                target_expr = loma_ir.Var(target, t=node.t)
                self.adj_accum_stmts += \
                    accum_deriv(var_to_differential(node, self.var_to_dvar),
                        target_expr, overwrite = False)
                return [accum_deriv(target_expr, self.adj, overwrite = True)]
            else:
                return [accum_deriv(var_to_differential(node, self.var_to_dvar),
                    self.adj, overwrite = False)]

        def mutate_const_float(self, node):
            return []

        def mutate_const_int(self, node):
            return []

        def mutate_array_access(self, node):
            if self.in_assign:
                target = f'_adj_{str(self.adj_count)}'
                self.adj_count += 1
                self.adj_declaration.append(loma_ir.Declare(target, t=node.t))
                target_expr = loma_ir.Var(target, t=node.t)
                self.adj_accum_stmts += \
                    accum_deriv(var_to_differential(node, self.var_to_dvar),
                        target_expr, overwrite = False)
                return [accum_deriv(target_expr, self.adj, overwrite = True)]
            else:
                return [accum_deriv(var_to_differential(node, self.var_to_dvar),
                    self.adj, overwrite = False)]

        def mutate_struct_access(self, node):
            if self.in_assign:
                target = f'_adj_{str(self.adj_count)}'
                self.adj_count += 1
                self.adj_declaration.append(loma_ir.Declare(target, t=node.t))
                target_expr = loma_ir.Var(target, t=node.t)
                self.adj_accum_stmts += \
                    accum_deriv(var_to_differential(node, self.var_to_dvar),
                        target_expr, overwrite = False)
                return [accum_deriv(target_expr, self.adj, overwrite = True)]
            else:
                return [accum_deriv(var_to_differential(node, self.var_to_dvar),
                    self.adj, overwrite = False)]

        def mutate_add(self, node):
            left = self.mutate_expr(node.left)
            right = self.mutate_expr(node.right)
            return left + right

        def mutate_sub(self, node):
            old_adj = self.adj
            left = self.mutate_expr(node.left)
            self.adj = loma_ir.BinaryOp(loma_ir.Sub(),
                loma_ir.ConstFloat(0.0), old_adj)
            right = self.mutate_expr(node.right)
            self.adj = old_adj
            return left + right

        def mutate_mul(self, node):
            # z = x * y
            # dz/dx = dz * y
            # dz/dy = dz * x
            old_adj = self.adj
            self.adj = loma_ir.BinaryOp(loma_ir.Mul(),
                node.right, old_adj)
            left = self.mutate_expr(node.left)
            self.adj = loma_ir.BinaryOp(loma_ir.Mul(),
                node.left, old_adj)
            right = self.mutate_expr(node.right)
            self.adj = old_adj
            return left + right

        def mutate_div(self, node):
            # z = x / y
            # dz/dx = dz / y
            # dz/dy = - dz * x / y^2
            old_adj = self.adj
            self.adj = loma_ir.BinaryOp(loma_ir.Div(),
                old_adj, node.right)
            left = self.mutate_expr(node.left)
            # - dz
            self.adj = loma_ir.BinaryOp(loma_ir.Sub(),
                loma_ir.ConstFloat(0.0), old_adj)
            # - dz * x
            self.adj = loma_ir.BinaryOp(loma_ir.Mul(),
                self.adj, node.left)
            # - dz * x / y^2
            self.adj = loma_ir.BinaryOp(loma_ir.Div(),
                self.adj, loma_ir.BinaryOp(loma_ir.Mul(), node.right, node.right))
            right = self.mutate_expr(node.right)
            self.adj = old_adj
            return left + right

        def mutate_call(self, node):
            match node.id:
                case 'sin':
                    assert len(node.args) == 1
                    old_adj = self.adj
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        loma_ir.Call(\
                            'cos',
                            node.args,
                            lineno = node.lineno,
                            t = node.t),
                        old_adj,
                        lineno = node.lineno)
                    ret = self.mutate_expr(node.args[0])
                    self.adj = old_adj
                    return ret
                case 'cos':
                    assert len(node.args) == 1
                    old_adj = self.adj
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Sub(),
                        loma_ir.ConstFloat(0.0),
                        loma_ir.BinaryOp(
                            loma_ir.Mul(),
                            loma_ir.Call(\
                                'sin',
                                node.args,
                                lineno = node.lineno,
                                t = node.t),
                            self.adj,
                            lineno = node.lineno),
                        lineno = node.lineno)
                    ret = self.mutate_expr(node.args[0]) 
                    self.adj = old_adj
                    return ret
                case 'sqrt':
                    assert len(node.args) == 1
                    # y = sqrt(x)
                    # dx = (1/2) * dy / y
                    old_adj = self.adj
                    sqrt = loma_ir.Call(\
                        'sqrt',
                        node.args,
                        lineno = node.lineno,
                        t = node.t)
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Mul(),
                        loma_ir.ConstFloat(0.5), self.adj,
                        lineno = node.lineno)
                    self.adj = loma_ir.BinaryOp(
                        loma_ir.Div(),
                        self.adj, sqrt,
                        lineno = node.lineno)
                    ret = self.mutate_expr(node.args[0])
                    self.adj = old_adj
                    return ret
                case 'pow':
                    assert len(node.args) == 2
                    # y = pow(x0, x1)
                    # dx0 = dy * x1 * pow(x0, x1 - 1)
                    # dx1 = dy * pow(x0, x1) * log(x0)
                    base_expr = node.args[0]
                    exp_expr = node.args[1]

                    old_adj = self.adj
                    # base term
                    self.adj = loma_ir.BinaryOp(\
                        loma_ir.Mul(),
                        self.adj, exp_expr,
                        lineno = node.lineno)
                    exp_minus_1 = loma_ir.BinaryOp(\
                        loma_ir.Sub(),
                        exp_expr, loma_ir.ConstFloat(1.0),
                        lineno = node.lineno)
                    pow_exp_minus_1 = loma_ir.Call(\
                        'pow',
                        [base_expr, exp_minus_1],
                        lineno = node.lineno,
                        t = node.t)
                    self.adj = loma_ir.BinaryOp(\
                        loma_ir.Mul(),
                        self.adj, pow_exp_minus_1,
                        lineno = node.lineno)
                    base_stmts = self.mutate_expr(base_expr)
                    self.adj = old_adj

                    # exp term
                    pow_expr = loma_ir.Call(\
                        'pow',
                        [base_expr, exp_expr],
                        lineno = node.lineno,
                        t = node.t)
                    self.adj = loma_ir.BinaryOp(\
                        loma_ir.Mul(),
                        self.adj, pow_expr,
                        lineno = node.lineno)
                    log = loma_ir.Call(\
                        'log',
                        [base_expr],
                        lineno = node.lineno)
                    self.adj = loma_ir.BinaryOp(\
                        loma_ir.Mul(),
                        self.adj, log,
                        lineno = node.lineno)
                    exp_stmts = self.mutate_expr(exp_expr)
                    self.adj = old_adj
                    return base_stmts + exp_stmts
                case 'exp':
                    assert len(node.args) == 1
                    exp = loma_ir.Call(\
                        'exp',
                        node.args,
                        lineno = node.lineno,
                        t = node.t)
                    old_adj = self.adj
                    self.adj = loma_ir.BinaryOp(\
                        loma_ir.Mul(),
                        self.adj, exp,
                        lineno = node.lineno)
                    ret = self.mutate_expr(node.args[0])
                    self.adj = old_adj
                    return ret
                case 'log':
                    assert len(node.args) == 1
                    old_adj = self.adj
                    self.adj = loma_ir.BinaryOp(\
                        loma_ir.Div(),
                        self.adj, node.args[0])
                    ret = self.mutate_expr(node.args[0])
                    self.adj = old_adj
                    return ret
                case 'int2float':
                    # don't propagate the derivatives
                    return []
                case 'float2int':
                    # don't propagate the derivatives
                    return []
                case 'atomic_add':
                    # z = z + x
                    # -> dx += dz
                    target = var_to_differential(\
                        node.args[1], self.var_to_dvar)
                    source = var_to_differential(\
                        node.args[0], self.var_to_dvar)
                    return accum_deriv(target, source, overwrite = False)
                case 'make__dfloat':
                    # z = make__dfloat(x, y)
                    old_adj = self.adj
                    self.adj = loma_ir.StructAccess(old_adj,
                        'val')
                    ret0 = self.mutate_expr(node.args[0])
                    self.adj = loma_ir.StructAccess(old_adj,
                        'dval')
                    ret1 = self.mutate_expr(node.args[1])
                    self.adj = old_adj
                    return ret0 + ret1
                case _: # user function
                    # f(x0, x1, ...) -> df(x0, _dx0, x1, _dx1, ...)
                    f = funcs[node.id]
                    new_args = []
                    stmts = []
                    for i, f_arg in enumerate(f.args):
                        arg_expr = node.args[i]
                        assert isinstance(arg_expr, loma_ir.Var)
                        if f_arg.i == loma_ir.In():
                            new_args.append(arg_expr)
                            new_args.append(loma_ir.Var(self.var_to_dvar[arg_expr.id], t = f_arg.t))
                        else:
                            assert f_arg.i == loma_ir.Out()
                            new_args.append(loma_ir.Var(self.var_to_dvar[arg_expr.id], t = f_arg.t))
                    if f.ret_type is not None:
                        new_args.append(self.adj)

                    stmts.append(loma_ir.CallStmt(loma_ir.Call(\
                        func_to_rev[node.id],
                        new_args,
                        t = None)))
                    return stmts

    return RevDiffMutator().mutate_function_def(func)
