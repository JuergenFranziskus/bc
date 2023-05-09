use crate::frontend::{
    expr_tree::{Expr, ExprKind, ExprTree, Function},
    symbols::{function::FunctionID, variable::VarID, Symbols},
    types::{pointer_type::PointerTypeID, IntType, RegularIntKind, SpecialIntKind, Type, Types},
};
use cir::instruction::Expr as CExpr;
use cir::variable::VarID as CVarID;
use cir::{
    function::FuncID, instruction::ConstValue, register::RegID, Builder, IntegerSize, Module,
    Type as CType, Types as CTypes,
};
use std::collections::HashMap;

pub struct TargetInfo {
    pub size_bits: u32,
    pub ptr_bits: u32,
}

pub struct CodeGen<'a> {
    target: TargetInfo,
    symbols: &'a Symbols<'a>,
    types: &'a Types,
    builder: Builder,
    funcs: HashMap<FunctionID, FuncID>,
    in_function: Option<FuncCtx>,
}
impl<'a> CodeGen<'a> {
    pub fn new(symbols: &'a Symbols<'a>, types: &'a Types, target: TargetInfo) -> Self {
        Self {
            target,
            symbols,
            types,
            builder: Builder::new(Module::new(), CTypes::new()),
            funcs: HashMap::new(),
            in_function: None,
        }
    }

    pub fn gen_code(mut self, tree: &ExprTree) -> (Module, CTypes) {
        for function in &tree.functions {
            self.register_function(function);
        }
        for function in &tree.functions {
            self.build_function(function);
        }

        self.builder.finish()
    }
    fn register_function(&mut self, f: &Function) {
        let func = &self.symbols[f.id];
        let name = mangle_name(func.name());
        let return_type = self.make_type(func.return_type());
        let ir_id = self.builder.create_function(name, return_type);
        self.funcs.insert(f.id, ir_id);
    }
    fn build_function(&mut self, f: &Function) {
        let ir_id = self.funcs[&f.id];
        self.enter_function(ir_id);
        self.register_vars(f);
        self.store_parameters(f);
        let ret = self.build_expr(&f.body);
        let ret = self.make_rvalue(ret);
        self.builder.do_return(ret);

        self.exit_function();
    }
    fn enter_function(&mut self, ir_id: FuncID) {
        self.in_function = Some(FuncCtx {
            vars: HashMap::new(),
            var_ptrs: HashMap::new(),
        });
        self.builder.start_function(ir_id);
    }
    fn register_vars(&mut self, f: &Function) {
        let mut vars: Vec<_> = self.symbols[f.id].variables().iter().copied().collect();
        vars.sort_by_key(|v| v.0); // Probably not necessary, just nice for deterministic order and stuff.

        for var in vars {
            let var_type = self.symbols[var].var_type();
            let var_type = self.make_type(var_type);
            let var_id = self.builder.add_variable(var_type);
            let var_ptr = self.builder.get_var_ptr(var_id);
            let function = self.in_function.as_mut().unwrap();
            function.vars.insert(var, var_id);
            function.var_ptrs.insert(var, var_ptr);
        }
    }
    fn store_parameters(&mut self, f: &Function) {
        let func = &self.symbols[f.id];
        for &param in func.parameters() {
            let ir_var = self.get_ir_var(param);
            let var_type = self.builder.module()[ir_var].var_type();
            let param_reg = self.builder.add_func_param(var_type);
            let var_ptr = self.get_var_ptr(param);
            self.builder.store(var_ptr, param_reg);
        }
    }
    fn exit_function(&mut self) {
        self.in_function = None;
        self.builder.unselect_block();
    }

    fn build_expr(&mut self, e: &Expr) -> Value {
        match &e.kind {
            ExprKind::Assign(a, b) => self.build_assign(a, b),
            ExprKind::Add(a, b) => self.build_add(a, b),
            ExprKind::Sub(a, b) => self.build_sub(a, b),
            ExprKind::Mul(a, b) => self.build_mul(a, b),
            ExprKind::Div(a, b) => self.build_div(a, b),
            ExprKind::ShiftLeft(a, by) => self.build_shift_left(a, by),
            ExprKind::ShiftRight(a, by) => self.build_shift_right(a, by),
            ExprKind::TestEqual(a, b) => self.build_test_equal(a, b),
            ExprKind::TestNotEqual(a, b) => self.build_test_not_equal(a, b),
            ExprKind::TestLess(a, b) => self.build_test_less(a, b),
            ExprKind::TestGreater(a, b) => self.build_test_greater(a, b),
            ExprKind::TestLessEqual(a, b) => self.build_test_less_equal(a, b),
            ExprKind::TestGreaterEqual(a, b) => self.build_test_greater_equal(a, b),
            ExprKind::And(a, b) => self.build_and(a, b),
            ExprKind::Or(a, b) => self.build_or(a, b),
            ExprKind::BitAnd(a, b) => self.build_bitand(a, b),
            ExprKind::BitOr(a, b) => self.build_bitor(a, b),
            ExprKind::BitXor(a, b) => self.build_bitxor(a, b),
            ExprKind::SliceLength(ptr) => self.build_slice_length(ptr),
            ExprKind::MakeSlice(ptr, size) => self.build_make_slice(ptr, size),

            ExprKind::Negate(a) => self.build_negate(a),
            ExprKind::Not(a) | ExprKind::BitNot(a) => self.build_not(a),

            ExprKind::AddrOf(lvalue) | ExprKind::AddrOfMut(lvalue) => self.build_addr_of(lvalue),
            ExprKind::Deref(pointer) => self.build_deref(pointer),
            ExprKind::VolatileStore(ptr, value) => self.build_volatile_store(ptr, value),

            ExprKind::Conversion(from) => self.build_conversion(from, e.expr_type),

            ExprKind::Paren(inner) => self.build_expr(inner),
            ExprKind::Call(func, args) => self.build_call(func, args),
            ExprKind::Index(array, index) => self.build_index(array, index),
            &ExprKind::TupleIndex(ref tuple, index) => self.build_tuple_index(tuple, index),

            ExprKind::If(c, then, els) => self.build_if(c, then, els, e.expr_type),
            ExprKind::While(c, then) => self.build_while(c, then),
            ExprKind::Block(statements, value) => self.build_block(statements, value),

            ExprKind::Tuple(values) => self.build_tuple(values),
            ExprKind::Array(values) => self.build_array(values),
            &ExprKind::ShortArray(ref value, length) => self.build_short_array(value, length),
            &ExprKind::Variable(id) => self.build_variable(id),
            &ExprKind::Function(id) => self.build_function_expr(id),
            &ExprKind::Decimal(value) => self.build_decimal(value),
            &ExprKind::Boolean(value) => self.build_boolean(value),
            &ExprKind::Unit => self.build_unit(),
        }
    }
    fn make_rvalue(&mut self, value: Value) -> CExpr {
        match value {
            Value::LValue(ptr, t) => self.builder.load(ptr, t).into(),
            Value::RValue(reg) => reg.into(),
        }
    }

    fn build_assign(&mut self, a: &Expr, b: &Expr) -> Value {
        let a = self.build_expr(a);
        let Value::LValue(ptr, _t) = a else { unreachable!() };
        let b = self.build_expr(b);
        let b = self.make_rvalue(b);

        self.builder.store(ptr, b);

        Value::RValue(().into())
    }
    fn build_add(&mut self, a: &Expr, b: &Expr) -> Value {
        let a = self.build_expr(a);
        let a = self.make_rvalue(a);
        let b = self.build_expr(b);
        let b = self.make_rvalue(b);

        let reg = self.builder.add(a, b);
        Value::RValue(reg.into())
    }
    fn build_sub(&mut self, a: &Expr, b: &Expr) -> Value {
        let a = self.build_expr(a);
        let a = self.make_rvalue(a);
        let b = self.build_expr(b);
        let b = self.make_rvalue(b);

        let reg = self.builder.sub(a, b);
        Value::RValue(reg.into())
    }
    fn build_mul(&mut self, a: &Expr, b: &Expr) -> Value {
        let a = self.build_expr(a);
        let a = self.make_rvalue(a);
        let b = self.build_expr(b);
        let b = self.make_rvalue(b);

        let reg = self.builder.mul(a, b);
        Value::RValue(reg.into())
    }
    fn build_div(&mut self, a: &Expr, b: &Expr) -> Value {
        let Type::Integer(int) = a.expr_type else { unreachable!() };
        let signed = int.signed();

        let a = self.build_expr(a);
        let a = self.make_rvalue(a);
        let b = self.build_expr(b);
        let b = self.make_rvalue(b);

        let reg = if signed {
            self.builder.idiv(a, b)
        } else {
            self.builder.udiv(a, b)
        };
        Value::RValue(reg.into())
    }
    fn build_shift_left(&mut self, a: &Expr, by: &Expr) -> Value {
        let a = self.build_expr(a);
        let a = self.make_rvalue(a);
        let by = self.build_expr(by);
        let by = self.make_rvalue(by);

        let reg = self.builder.shl(a, by);
        Value::RValue(reg.into())
    }
    fn build_shift_right(&mut self, a: &Expr, by: &Expr) -> Value {
        let Type::Integer(int) = a.expr_type else { unreachable!() };
        let signed = int.signed();

        let a = self.build_expr(a);
        let a = self.make_rvalue(a);
        let by = self.build_expr(by);
        let by = self.make_rvalue(by);

        let reg = if signed {
            self.builder.sar(a, by)
        } else {
            self.builder.shr(a, by)
        };
        Value::RValue(reg.into())
    }
    fn build_test_equal(&mut self, a: &Expr, b: &Expr) -> Value {
        let a = self.build_expr(a);
        let a = self.make_rvalue(a);
        let b = self.build_expr(b);
        let b = self.make_rvalue(b);

        let reg = self.builder.test_equal(a, b);
        Value::RValue(reg.into())
    }
    fn build_test_not_equal(&mut self, a: &Expr, b: &Expr) -> Value {
        let a = self.build_expr(a);
        let a = self.make_rvalue(a);
        let b = self.build_expr(b);
        let b = self.make_rvalue(b);

        let reg = self.builder.test_not_equal(a, b);
        Value::RValue(reg.into())
    }
    fn build_test_less(&mut self, a: &Expr, b: &Expr) -> Value {
        let Type::Integer(int) = a.expr_type else { unreachable!() };
        let signed = int.signed();

        let a = self.build_expr(a);
        let a = self.make_rvalue(a);
        let b = self.build_expr(b);
        let b = self.make_rvalue(b);

        let reg = if signed {
            self.builder.test_less(a, b)
        } else {
            self.builder.test_below(a, b)
        };
        Value::RValue(reg.into())
    }
    fn build_test_greater(&mut self, a: &Expr, b: &Expr) -> Value {
        let Type::Integer(int) = a.expr_type else { unreachable!() };
        let signed = int.signed();

        let a = self.build_expr(a);
        let a = self.make_rvalue(a);
        let b = self.build_expr(b);
        let b = self.make_rvalue(b);

        let reg = if signed {
            self.builder.test_greater(a, b)
        } else {
            self.builder.test_above(a, b)
        };
        Value::RValue(reg.into())
    }
    fn build_test_less_equal(&mut self, a: &Expr, b: &Expr) -> Value {
        let Type::Integer(int) = a.expr_type else { unreachable!() };
        let signed = int.signed();

        let a = self.build_expr(a);
        let a = self.make_rvalue(a);
        let b = self.build_expr(b);
        let b = self.make_rvalue(b);

        let reg = if signed {
            self.builder.test_less_equal(a, b)
        } else {
            self.builder.test_below_equal(a, b)
        };
        Value::RValue(reg.into())
    }
    fn build_test_greater_equal(&mut self, a: &Expr, b: &Expr) -> Value {
        let Type::Integer(int) = a.expr_type else { unreachable!() };
        let signed = int.signed();

        let a = self.build_expr(a);
        let a = self.make_rvalue(a);
        let b = self.build_expr(b);
        let b = self.make_rvalue(b);

        let reg = if signed {
            self.builder.test_greater_equal(a, b)
        } else {
            self.builder.test_above_equal(a, b)
        };
        Value::RValue(reg.into())
    }
    fn build_and(&mut self, a: &Expr, b: &Expr) -> Value {
        let true_branch = self.builder.add_block();
        let end_branch = self.builder.add_block();

        let a = self.build_expr(a);
        let a = self.make_rvalue(a);
        self.builder.branch(a, true_branch, (end_branch, false));

        self.builder.select_block(true_branch);
        let b = self.build_expr(b);
        let b = self.make_rvalue(b);
        self.builder.jump((end_branch, b));

        self.builder.select_block(end_branch);
        let value = self.builder.add_block_param(IntegerSize::make(1));
        Value::RValue(value.into())
    }
    fn build_or(&mut self, a: &Expr, b: &Expr) -> Value {
        let false_branch = self.builder.add_block();
        let end_branch = self.builder.add_block();

        let a = self.build_expr(a);
        let a = self.make_rvalue(a);
        self.builder.branch(a, (end_branch, true), false_branch);

        self.builder.select_block(false_branch);
        let b = self.build_expr(b);
        let b = self.make_rvalue(b);
        self.builder.jump((end_branch, b));

        self.builder.select_block(end_branch);
        let value = self.builder.add_block_param(IntegerSize::make(1));
        Value::RValue(value.into())
    }
    fn build_bitand(&mut self, a: &Expr, b: &Expr) -> Value {
        let a = self.build_expr(a);
        let a = self.make_rvalue(a);
        let b = self.build_expr(b);
        let b = self.make_rvalue(b);

        let reg = self.builder.or(a, b);
        Value::RValue(reg.into())
    }
    fn build_bitor(&mut self, a: &Expr, b: &Expr) -> Value {
        let a = self.build_expr(a);
        let a = self.make_rvalue(a);
        let b = self.build_expr(b);
        let b = self.make_rvalue(b);

        let reg = self.builder.or(a, b);
        Value::RValue(reg.into())
    }
    fn build_bitxor(&mut self, a: &Expr, b: &Expr) -> Value {
        let a = self.build_expr(a);
        let a = self.make_rvalue(a);
        let b = self.build_expr(b);
        let b = self.make_rvalue(b);

        let reg = self.builder.xor(a, b);
        Value::RValue(reg.into())
    }
    fn build_slice_length(&mut self, slice: &Expr) -> Value {
        let slice = self.build_expr(slice);
        let slice = self.make_rvalue(slice);

        let reg = self.builder.get_member(slice, 1);
        Value::RValue(reg.into())
    }
    fn build_make_slice(&mut self, ptr: &Expr, size: &Expr) -> Value {
        let ptr = self.build_expr(ptr);
        let size = self.build_expr(size);
        let ptr = self.make_rvalue(ptr);
        let size = self.make_rvalue(size);

        let structure = CExpr::Struct(vec![ptr, size]);
        Value::RValue(structure)
    }
    fn build_negate(&mut self, a: &Expr) -> Value {
        let a = self.build_expr(a);
        let a = self.make_rvalue(a);
        let reg = self.builder.neg(a);
        Value::RValue(reg.into())
    }
    fn build_not(&mut self, a: &Expr) -> Value {
        let a = self.build_expr(a);
        let a = self.make_rvalue(a);
        let reg = self.builder.not(a);
        Value::RValue(reg.into())
    }
    fn build_addr_of(&mut self, lvalue: &Expr) -> Value {
        let lvalue = self.build_expr(lvalue);
        let Value::LValue(ptr, _t) = lvalue else { unreachable!() };
        Value::RValue(ptr.into())
    }
    fn build_deref(&mut self, pointer: &Expr) -> Value {
        let Type::Pointer(id) = pointer.expr_type else { unreachable!() };
        let pointee = self.types[id].pointee();
        let pointee = self.make_type(pointee);

        let pointer = self.build_expr(pointer);
        let pointer = self.make_rvalue(pointer);

        let CExpr::Register(reg) = pointer else { panic!() };

        Value::LValue(reg, pointee)
    }
    fn build_volatile_store(&mut self, pointer: &Expr, value: &Expr) -> Value {
        let ptr = self.build_expr(pointer);
        let ptr = self.make_rvalue(ptr);
        let value = self.build_expr(value);
        let value = self.make_rvalue(value);

        let CExpr::Register(reg) = ptr else { panic!() };
        self.builder.store_volatile(reg, value);

        Value::RValue(().into())
    }
    fn build_conversion(&mut self, from_value: &Expr, to: Type) -> Value {
        let from = from_value.expr_type;

        let from_value = self.build_expr(from_value);
        let from_value = self.make_rvalue(from_value);

        use Type::*;
        let reg = match (from, to) {
            (Pointer(f), Pointer(t)) => self.build_ptr_ptr_conversion(from_value, f, t),
            (Pointer(_), Integer(t)) => self
                .builder
                .ptr_to_int(from_value, self.make_int_type(t))
                .into(),
            (Integer(_), Pointer(_)) => self.builder.int_to_ptr(from_value).into(),
            (Integer(f), Integer(t)) => self.build_int_int_conversion(from_value, f, t),
            (Literal(value), Integer(t)) => {
                let t_size = self.make_int_type(t);
                CExpr::Constant(ConstValue::Integer(value, t_size))
            }
            (Integer(f), Bool) => {
                let f_size = self.make_int_type(f);
                let zero = CExpr::Constant(ConstValue::Integer(0, f_size));
                self.builder.test_not_equal(from_value, zero).into()
            }
            (Bool, Integer(t)) => {
                let t_size = self.make_int_type(t);
                self.builder.zext(from_value, t_size).into()
            }
            _ => panic!("Cannot convert from {from:?} to {to:?}"),
        };

        Value::RValue(reg)
    }
    fn build_ptr_ptr_conversion(
        &mut self,
        from_value: CExpr,
        from: PointerTypeID,
        to: PointerTypeID,
    ) -> CExpr {
        let from_pointee = self.types[from].pointee();
        let to_pointee = self.types[to].pointee();
        let CExpr::Register(ptr) = from_value else { panic!() };

        use Type::*;
        match (from_pointee, to_pointee) {
            (Slice(f), Slice(t)) => {
                let slice_struct = ptr;
                let ptr = self.builder.get_member(slice_struct, 0);
                let mut len = self.builder.get_member(slice_struct, 1);

                let f_element = self.types[f].member();
                let t_element = self.types[t].member();

                if !f_element.is_void() {
                    let f_type = self.make_type(f_element.into());
                    let size = CExpr::Constant(ConstValue::SizeOf(
                        f_type,
                        IntegerSize::make(self.target.size_bits),
                    ));
                    len = self.builder.mul(len, size);
                }
                if !t_element.is_void() {
                    let t_type = self.make_type(t_element.into());
                    let size = CExpr::Constant(ConstValue::SizeOf(
                        t_type,
                        IntegerSize::make(self.target.size_bits),
                    ));
                    len = self.builder.udiv(len, size);
                }

                CExpr::Struct(vec![ptr.into(), len.into()])
            }
            (Slice(_), _) => todo!(),
            (Array(f), Slice(t)) => {
                let len = self.types[f].length();
                let mut len = CExpr::Constant(ConstValue::Integer(
                    len.into(),
                    IntegerSize::make(self.target.size_bits),
                ));

                let f_element = self.types[f].member();
                let t_element = self.types[t].member();

                if !f_element.is_void() {
                    let f_type = self.make_type(f_element.into());
                    let size = CExpr::Constant(ConstValue::SizeOf(
                        f_type,
                        IntegerSize::make(self.target.size_bits),
                    ));
                    len = self.builder.mul(len, size).into();
                }
                if !t_element.is_void() {
                    let t_type = self.make_type(t_element.into());
                    let size = CExpr::Constant(ConstValue::SizeOf(
                        t_type,
                        IntegerSize::make(self.target.size_bits),
                    ));
                    len = self.builder.udiv(len, size).into();
                }

                CExpr::Struct(vec![ptr.into(), len])
            }
            (_, Slice(_)) => unreachable!(),
            _ => from_value,
        }
    }
    fn build_int_int_conversion(&mut self, from_value: CExpr, from: IntType, to: IntType) -> CExpr {
        let to_signed = to.signed();

        let from_size = self.make_int_type(from);
        let to_size = self.make_int_type(to);

        if from_size == to_size {
            from_value
        } else if from_size > to_size {
            self.builder.trunc(from_value, to_size).into()
        } else if to_signed {
            self.builder.sext(from_value, to_size).into()
        } else {
            self.builder.zext(from_value, to_size).into()
        }
    }
    fn build_call(&mut self, func: &Expr, args: &[Expr]) -> Value {
        let args = args
            .iter()
            .map(|a| {
                let a = self.build_expr(a);
                self.make_rvalue(a)
            })
            .collect();

        if let ExprKind::Function(id) = func.kind {
            let func = self.funcs[&id];
            let reg = self.builder.call(func, args);
            Value::RValue(reg.into())
        } else {
            let Type::Function(id) = func.expr_type else { panic!() };
            let return_type = self.types[id].return_type();
            let return_type = self.make_type(return_type);
            let func = self.build_expr(func);
            let func = self.make_rvalue(func);
            let CExpr::Register(ptr) = func else { panic!() };

            let reg = self
                .builder
                .call_ptr(ptr, return_type, args, Default::default(), false);
            Value::RValue(reg.into())
        }
    }
    fn build_index(&mut self, array: &Expr, index: &Expr) -> Value {
        let array_value = self.build_expr(array);
        let index = self.build_expr(index);
        let index = self.make_rvalue(index);
        if let Type::Array(_) = array.expr_type {
            match array_value {
                Value::RValue(arr) => {
                    let reg = self.builder.get_element(arr, index);
                    Value::RValue(reg.into())
                }
                Value::LValue(arr, t) => {
                    let CType::Array(id) = t else { unreachable!() };
                    let element = self.builder.types()[id].member();
                    let reg = self.builder.get_element_ptr(arr, index, element);
                    Value::LValue(reg, element)
                }
            }
        } else {
            let Type::Pointer(id) = array.expr_type else { unreachable!() };
            let pointee = self.types[id].pointee();
            let array_value = self.make_rvalue(array_value);
            let ptr = array_value;

            if let Type::Array(id) = pointee {
                let CExpr::Register(ptr) = ptr else { unreachable!() };
                let element = self.types[id].member();
                let element = self.make_type(element);
                let reg = self.builder.get_element_ptr(ptr, index, element);
                Value::LValue(reg, element)
            } else {
                let Type::Slice(id) = pointee else { unreachable!() };
                let ptr_struct = ptr;
                let ptr = self.builder.get_member(ptr_struct, 0);
                let element = self.types[id].member();
                let element = self.make_type(element);
                let reg = self.builder.get_element_ptr(ptr, index, element);
                Value::LValue(reg, element)
            }
        }
    }
    fn build_tuple_index(&mut self, tuple: &Expr, index: u64) -> Value {
        let tuple = self.build_expr(tuple);

        match tuple {
            Value::RValue(structure) => {
                let reg = self.builder.get_member(structure, index as u32);
                Value::RValue(reg.into())
            }
            Value::LValue(ptr, structure_type) => {
                let CType::Struct(id) = structure_type else { panic!() };
                let member_type = self.builder.types()[id].members()[index as usize];
                let ptr = self.builder.get_member_ptr(ptr, index as u32, id);
                Value::LValue(ptr, member_type)
            }
        }
    }
    fn build_if(&mut self, c: &Expr, then: &Expr, els: &Expr, expr_type: Type) -> Value {
        let then_branch = self.builder.add_block();
        let else_branch = self.builder.add_block();
        let end = self.builder.add_block();

        let c = self.build_expr(c);
        let c = self.make_rvalue(c);
        self.builder.branch(c, then_branch, else_branch);

        self.builder.select_block(then_branch);
        let value = self.build_expr(then);
        let value = self.make_rvalue(value);
        self.builder.jump((end, value));

        self.builder.select_block(else_branch);
        let value = self.build_expr(els);
        let value = self.make_rvalue(value);
        self.builder.jump((end, value));

        self.builder.select_block(end);
        let val_type = self.make_type(expr_type);
        let value = self.builder.add_block_param(val_type);
        Value::RValue(value.into())
    }
    fn build_while(&mut self, c: &Expr, then: &Expr) -> Value {
        let header = self.builder.add_block();
        let body = self.builder.add_block();
        let end = self.builder.add_block();

        self.builder.jump(header);

        self.builder.select_block(header);
        let c = self.build_expr(c);
        let c = self.make_rvalue(c);
        self.builder.branch(c, body, end);

        self.builder.select_block(body);
        self.build_expr(then);
        self.builder.jump(header);

        self.builder.select_block(end);
        Value::RValue(().into())
    }
    fn build_block(&mut self, statements: &[Expr], value: &Expr) -> Value {
        for statement in statements {
            self.build_expr(statement);
        }
        let value = self.build_expr(value);
        let value = self.make_rvalue(value);
        Value::RValue(value)
    }
    fn build_tuple(&mut self, values: &[Expr]) -> Value {
        let values = values
            .iter()
            .map(|v| {
                let v = self.build_expr(v);
                self.make_rvalue(v)
            })
            .collect();

        Value::RValue(CExpr::Struct(values))
    }
    fn build_array(&mut self, values: &[Expr]) -> Value {
        let values = values
            .iter()
            .map(|v| {
                let value = self.build_expr(v);
                self.make_rvalue(value)
            })
            .collect();

        Value::RValue(CExpr::Array(values))
    }
    fn build_short_array(&mut self, value: &Expr, length: u64) -> Value {
        let value = self.build_expr(value);
        let value = self.make_rvalue(value);
        Value::RValue(CExpr::ShortArray(value.into(), length))
    }
    fn build_variable(&mut self, id: VarID) -> Value {
        let ir_id = self.get_ir_var(id);
        let var_type = self.builder.module()[ir_id].var_type();
        let ptr = self.get_var_ptr(id);
        Value::LValue(ptr, var_type)
    }
    fn build_function_expr(&mut self, f: FunctionID) -> Value {
        let ir_id = self.funcs[&f];
        let reg = self.builder.get_func_ptr(ir_id);
        Value::RValue(reg.into())
    }
    fn build_decimal(&mut self, value: i128) -> Value {
        Value::RValue(CExpr::Constant(ConstValue::Integer(
            value,
            IntegerSize::make(64),
        )))
    }
    fn build_boolean(&mut self, value: bool) -> Value {
        Value::RValue(CExpr::Constant(ConstValue::Integer(
            value as i128,
            IntegerSize::make(1),
        )))
    }
    fn build_unit(&mut self) -> Value {
        Value::RValue(().into())
    }

    fn get_ir_var(&self, var: VarID) -> CVarID {
        self.in_function.as_ref().unwrap().vars[&var]
    }
    fn get_var_ptr(&self, var: VarID) -> RegID {
        self.in_function.as_ref().unwrap().var_ptrs[&var]
    }

    fn make_type(&mut self, t: Type) -> CType {
        match t {
            Type::Void => panic!(),
            Type::Unit => CType::Unit,
            Type::Bool => CType::integer(1),
            Type::Integer(int) => self.make_int_type(int).into(),
            Type::Literal(_) => panic!(),
            Type::Slice(_) => panic!(),
            Type::Function(_) => CType::Pointer,
            Type::Array(id) => {
                let element = self.types[id].member();
                let length = self.types[id].length();
                let element = self.make_type(element);
                let ctype = self.builder.types_mut().make_array(element, length);
                ctype.into()
            }
            Type::Tuple(id) => {
                let members = self.types[id].members();
                let members = members.iter().map(|&m| self.make_type(m)).collect();
                self.builder.types_mut().make_struct(members).into()
            }
            Type::Pointer(id) => {
                let pointee = self.types[id].pointee();
                match pointee {
                    Type::Slice(_) => self
                        .builder
                        .types_mut()
                        .make_struct(vec![CType::Pointer, CType::integer(self.target.size_bits)])
                        .into(),
                    _ => CType::Pointer,
                }
            }
        }
    }
    fn make_int_type(&self, t: IntType) -> IntegerSize {
        match t {
            IntType::Special(SpecialIntKind::Size, _) => IntegerSize::make(self.target.size_bits),
            IntType::Special(SpecialIntKind::Ptr, _) => IntegerSize::make(self.target.ptr_bits),
            IntType::Regular(RegularIntKind::Byte, _) => IntegerSize::make(8),
            IntType::Regular(RegularIntKind::Short, _) => IntegerSize::make(16),
            IntType::Regular(RegularIntKind::Int, _) => IntegerSize::make(32),
            IntType::Regular(RegularIntKind::Long, _) => IntegerSize::make(64),
        }
    }
}

fn mangle_name(name: &str) -> String {
    format!("_BC{name}")
}

enum Value {
    RValue(CExpr),
    LValue(RegID, CType),
}

struct FuncCtx {
    vars: HashMap<VarID, CVarID>,
    var_ptrs: HashMap<VarID, RegID>,
}
