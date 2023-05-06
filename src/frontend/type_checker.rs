use super::ast::{
    BinaryOp, Expr, ExprKind, Function, IntSuffix, Intrinsic, PrefixOp, TypeExpr, TypeExprKind,
};
use super::expr_tree::{Expr as EExpr, ExprKind as EExprKind, Function as EFunction};
use super::symbols::function::FunctionID;
use super::symbols::value_kind::ValueKind;
use super::symbols::variable::VarID;
use super::types::pointer_type::PointerTypeID;
use super::types::{IntType, Type};
use super::{ast::Ast, expr_tree::ExprTree, symbols::Symbols, types::Types};
use crate::frontend::types::{RegularIntKind, SpecialIntKind};
use std::collections::HashMap;

pub struct TypeChecker<'a> {
    symbols: Symbols<'a>,
    types: Types,
    scopes: Vec<Scope<'a>>,
    in_function: Option<FunctionID>,
}
impl<'a> TypeChecker<'a> {
    pub fn new() -> Self {
        Self {
            symbols: Symbols::new(),
            types: Types::new(),
            scopes: Vec::new(),
            in_function: None,
        }
    }

    pub fn type_check(mut self, ast: &Ast<'a>) -> (ExprTree, Symbols<'a>, Types) {
        let fids = self.collect_global_symbols(ast);
        let functions = self.check_functions(ast, fids);

        let tree = ExprTree { functions };
        (tree, self.symbols, self.types)
    }

    fn collect_global_symbols(&mut self, ast: &Ast<'a>) -> Vec<FunctionID> {
        let mut ids = Vec::with_capacity(ast.functions.len());

        for function in &ast.functions {
            let ret_type = function
                .return_type
                .as_ref()
                .map(|e| self.check_type_expr(e))
                .unwrap_or(Type::Unit);
            let fid = self.symbols.add_function(function.name, ret_type);
            for parameter in &function.parameters {
                let p_type = self.check_type_expr(&parameter.param_type);
                self.symbols
                    .add_parameter(fid, parameter.name, p_type, parameter.mutable);
            }
            ids.push(fid);
        }

        ids
    }
    fn check_functions(&mut self, ast: &Ast<'a>, fids: Vec<FunctionID>) -> Vec<EFunction> {
        ast.functions
            .iter()
            .zip(fids)
            .map(|(f, fid)| {
                let ret_type = self.symbols[fid].return_type();
                self.in_function = Some(fid);
                self.enter_scope();
                self.register_parameters(f, fid);
                let body = self.check_expr(&f.body);
                let body = self.coerce(body, ret_type);
                self.leave_scope();
                self.in_function = None;

                EFunction { id: fid, body }
            })
            .collect()
    }
    fn register_parameters(&mut self, f: &Function<'a>, fid: FunctionID) {
        for (i, param) in f.parameters.iter().enumerate() {
            let pid = self.symbols[fid].parameters()[i];
            self.register(param.name, pid);
        }
    }
    fn check_expr(&mut self, e: &Expr<'a>) -> EExpr {
        match &e.kind {
            ExprKind::Paren(inner) => self.check_expr(inner),
            ExprKind::SliceLength(slice) => self.check_slice_len(slice),
            ExprKind::MakeSlice(ptr, len) => self.check_make_slice(ptr, len),
            &ExprKind::TupleIndex(ref tuple, len) => self.check_tuple_index(tuple, len),
            ExprKind::Index(arr, index) => self.check_index(arr, index),
            ExprKind::Call(func, args) => self.check_call(func, args),
            &ExprKind::Intrinsic(intrinsic, ref args) => self.check_intrinsic(intrinsic, args),
            &ExprKind::Declaration(name, mutable, ref t, ref init) => {
                self.check_declaration(name, mutable, t.as_deref(), init)
            }
            ExprKind::BitCast(a, to) => self.check_bitcast(a, to),
            ExprKind::Cast(a, to) => self.check_cast(a, to),
            ExprKind::If(c, then, els) => self.check_if(c, then, els.as_deref()),
            ExprKind::While(c, body) => self.check_while(c, body),
            ExprKind::Block(members, value) => self.check_block(members, value.as_deref()),
            &ExprKind::Binary(op, ref a, ref b) => self.check_binary_op(op, a, b),
            &ExprKind::Prefix(op, ref a) => self.check_prefix_op(op, a),
            ExprKind::Tuple(members) => self.check_tuple(members),
            ExprKind::ShortArray(init, length) => self.check_short_array(init, length),
            ExprKind::Array(members) => self.check_array(members),
            &ExprKind::Named(name) => self.check_named(name),
            &ExprKind::Decimal(value, suffix) => self.check_decimal(value, suffix),
            &ExprKind::Bool(value) => self.check_bool(value),
        }
    }

    fn check_slice_len(&mut self, slice: &Expr<'a>) -> EExpr {
        let slice = self.check_expr(slice);

        let Type::Pointer(pid) = slice.expr_type else { panic!() };
        let pointee = self.types[pid].pointee();
        let Type::Slice(_) = pointee else { panic!() };

        EExpr {
            expr_type: Type::Integer(IntType::usize()),
            value_kind: ValueKind::RValue,
            kind: EExprKind::SliceLength(slice.into()),
        }
    }
    fn check_make_slice(&mut self, ptr: &Expr<'a>, length: &Expr<'a>) -> EExpr {
        let ptr = self.check_expr(ptr);
        let length = self.check_expr(length);
        let length = self.coerce(length, IntType::usize());

        let Type::Pointer(id) = ptr.expr_type else { panic!() };
        let mutable = self.types[id].mutable();
        let pointee = self.types[id].pointee();
        assert!(pointee.is_sized() || pointee.is_void());

        let formed_slice_type = self.types.add_slice_type(pointee);
        let formed_ptr_type = self
            .types
            .add_pointer_type(mutable, formed_slice_type.into());

        EExpr {
            expr_type: formed_ptr_type.into(),
            value_kind: ValueKind::RValue,
            kind: EExprKind::MakeSlice(ptr.into(), length.into()),
        }
    }
    fn check_tuple_index(&mut self, tuple: &Expr<'a>, index: u64) -> EExpr {
        let tuple = self.check_expr(tuple);
        let Type::Tuple(id) = tuple.expr_type else { panic!() };
        let member_amount = self.types[id].members().len() as u64;
        assert!(index < member_amount);
        let member = self.types[id].members()[index as usize];

        EExpr {
            expr_type: member,
            value_kind: tuple.value_kind,
            kind: EExprKind::TupleIndex(tuple.into(), index),
        }
    }
    fn check_index(&mut self, arr: &Expr<'a>, index: &Expr<'a>) -> EExpr {
        let arr = self.check_expr(arr);
        let index = self.check_expr(index);
        let index = self.make_literal_concrete(index);
        assert!(index.expr_type.is_integer());

        let expr_type;
        let value_kind;

        match arr.expr_type {
            Type::Array(id) => {
                expr_type = self.types[id].member();
                value_kind = arr.value_kind;
            }
            Type::Pointer(id) => {
                let pointee = self.types[id].pointee();
                let mutable = self.types[id].mutable();
                value_kind = ValueKind::LValue(mutable);

                match pointee {
                    Type::Array(id) => {
                        let member = self.types[id].member();
                        expr_type = member;
                    }
                    Type::Slice(id) => {
                        let member = self.types[id].member();
                        assert!(!member.is_void());
                        expr_type = member;
                    }
                    _ => panic!(),
                }
            }
            _ => panic!(),
        }

        EExpr {
            expr_type,
            value_kind,
            kind: EExprKind::Index(arr.into(), index.into()),
        }
    }
    fn check_call(&mut self, func: &Expr<'a>, args: &[Expr<'a>]) -> EExpr {
        let func = self.check_expr(func);
        let Type::Function(fid) = func.expr_type else { panic!() };
        let ret_t = self.types[fid].return_type();
        let args = args
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let a = self.check_expr(a);
                let param_t = self.types[fid].parameter_types()[i];
                let a = self.coerce(a, param_t);
                a
            })
            .collect();

        EExpr {
            expr_type: ret_t,
            value_kind: ValueKind::RValue,
            kind: EExprKind::Call(func.into(), args),
        }
    }
    fn check_intrinsic(&mut self, intrinsic: Intrinsic, args: &[Expr<'a>]) -> EExpr {
        let args: Vec<_> = args.iter().map(|a| self.check_expr(a)).collect();
        match intrinsic {
            Intrinsic::VolatileStore => self.check_volatile_store(args),
        }
    }
    fn check_volatile_store(&mut self, mut args: Vec<EExpr>) -> EExpr {
        assert_eq!(args.len(), 2);
        let value = args.pop().unwrap();
        let ptr = args.pop().unwrap();

        let Type::Pointer(id) = ptr.expr_type else { panic!() };
        assert!(self.types[id].mutable());
        let pointee = self.types[id].pointee();
        assert!(pointee.is_sized());
        let value = self.coerce(value, pointee);

        EExpr {
            expr_type: Type::Unit,
            value_kind: ValueKind::RValue,
            kind: EExprKind::VolatileStore(ptr.into(), value.into()),
        }
    }
    fn check_declaration(
        &mut self,
        name: &'a str,
        mutable: bool,
        t: Option<&TypeExpr>,
        init: &Expr<'a>,
    ) -> EExpr {
        let mut init = self.check_expr(init);

        let var = if let Some(t) = t {
            let init_t = self.check_type_expr(t);
            init = self.coerce(init, init_t);
            let vid = self
                .symbols
                .add_var(self.in_function.unwrap(), name, init_t, mutable);
            self.register(name, vid);
            EExpr {
                expr_type: init_t,
                value_kind: ValueKind::LValue(true),
                kind: EExprKind::Variable(vid),
            }
        } else {
            init = self.make_literal_concrete(init);
            let vid =
                self.symbols
                    .add_var(self.in_function.unwrap(), name, init.expr_type, mutable);
            self.register(name, vid);
            EExpr {
                expr_type: init.expr_type,
                value_kind: ValueKind::LValue(true),
                kind: EExprKind::Variable(vid),
            }
        };

        assert!(init.expr_type.is_sized());

        EExpr {
            expr_type: Type::Unit,
            value_kind: ValueKind::RValue,
            kind: EExprKind::Assign(var.into(), init.into()),
        }
    }
    fn check_bitcast(&mut self, a: &Expr<'a>, to: &TypeExpr) -> EExpr {
        let to = self.check_type_expr(to);
        let a = self.check_expr(a);
        let a = self.make_literal_concrete(a);

        assert!(self.can_bitcast(a.expr_type, to));

        EExpr {
            expr_type: to,
            value_kind: ValueKind::RValue,
            kind: EExprKind::Conversion(a.into()),
        }
    }
    fn can_bitcast(&self, from: Type, to: Type) -> bool {
        if self.can_coerce(from, to) {
            return true;
        }

        use IntType::*;
        use SpecialIntKind::*;
        use Type::*;
        match (from, to) {
            (a, b) if a == b => true,
            (Integer(Regular(a, _)), Integer(Regular(b, _))) => a == b,
            (Integer(Special(a, _)), Integer(Special(b, _))) => a == b,
            (Integer(Special(Ptr, _)), Pointer(_)) | (Pointer(_), Integer(Special(Ptr, _))) => true,
            _ => false,
        }
    }
    fn check_cast(&mut self, a: &Expr<'a>, to: &TypeExpr) -> EExpr {
        let to = self.check_type_expr(to);
        let a = self.check_expr(a);

        assert!(self.can_cast(a.expr_type, to));

        EExpr {
            expr_type: to,
            value_kind: ValueKind::RValue,
            kind: EExprKind::Conversion(a.into()),
        }
    }
    fn can_cast(&self, from: Type, to: Type) -> bool {
        if self.can_bitcast(from, to) {
            return true;
        }

        use Type::*;
        match (from, to) {
            (a, b) if a == b => true,
            (Pointer(f), Pointer(t)) => {
                let fp = self.types[f].pointee();
                let tp = self.types[t].pointee();
                match (fp, tp) {
                    (Slice(_), Slice(_)) => true,
                    (_, Slice(_)) => false,
                    _ => true,
                }
            }
            (Pointer(_), Integer(_)) => true,
            (Integer(_), Pointer(id)) => {
                let pointee = self.types[id].pointee();
                !matches!(pointee, Type::Slice(_))
            }
            (Integer(_), Integer(_)) => true,
            (Literal(_), Integer(_)) => true,
            (Integer(_), Bool) => true,
            (Bool, Integer(_)) => true,

            _ => false,
        }
    }
    fn check_if(&mut self, c: &Expr<'a>, then: &Expr<'a>, els: Option<&Expr<'a>>) -> EExpr {
        let c = self.check_expr(c);
        let c = self.coerce(c, Type::Bool);
        let then = self.check_expr(then);
        let els = els.map(|e| self.check_expr(e)).unwrap_or(EExpr::unit());
        let els = self.coerce(els, then.expr_type);

        EExpr {
            expr_type: then.expr_type,
            value_kind: ValueKind::RValue,
            kind: EExprKind::If(c.into(), then.into(), els.into()),
        }
    }
    fn check_while(&mut self, c: &Expr<'a>, body: &Expr<'a>) -> EExpr {
        let c = self.check_expr(c);
        let c = self.coerce(c, Type::Bool);
        let body = self.check_expr(body);

        EExpr {
            expr_type: Type::Unit,
            value_kind: ValueKind::RValue,
            kind: EExprKind::While(c.into(), body.into()),
        }
    }
    fn check_block(&mut self, members: &[Expr<'a>], value: Option<&Expr<'a>>) -> EExpr {
        self.enter_scope();

        let members = members.iter().map(|m| self.check_expr(m)).collect();
        let value = value.map(|v| self.check_expr(v)).unwrap_or(EExpr::unit());
        let value = self.make_literal_concrete(value);
        assert!(value.expr_type.is_sized());

        self.leave_scope();

        EExpr {
            expr_type: value.expr_type,
            value_kind: ValueKind::RValue,
            kind: EExprKind::Block(members, value.into()),
        }
    }
    fn check_binary_op(&mut self, op: BinaryOp, a: &Expr<'a>, b: &Expr<'a>) -> EExpr {
        let a = self.check_expr(a);
        let b = self.check_expr(b);

        use BinaryOp::*;
        match op {
            Assign => self.check_assignment(a, b),
            Add => self.check_add(a, b),
            Sub => self.check_sub(a, b),
            Mul => self.check_mul(a, b),
            Div => self.check_div(a, b),
            ShiftLeft => self.check_shift_left(a, b),
            ShiftRight => self.check_shift_right(a, b),
            TestEqual => self.check_test_equal(a, b),
            TestNotEqual => self.check_test_not_equal(a, b),
            TestGreater => self.check_test_greater(a, b),
            TestGreaterEqual => self.check_test_greater_equal(a, b),
            TestLess => self.check_test_less(a, b),
            TestLessEqual => self.check_test_less_equal(a, b),
            BitAnd => self.check_bitand(a, b),
            BitOr => self.check_bitor(a, b),
            BitXor => self.check_bitxor(a, b),
            And => self.check_and(a, b),
            Or => self.check_or(a, b),
        }
    }
    fn check_assignment(&self, a: EExpr, b: EExpr) -> EExpr {
        assert!(
            matches!(a.value_kind, ValueKind::LValue(true)),
            "Cannot assign to immutable LValue"
        );
        let b = self.coerce(b, a.expr_type);
        EExpr {
            expr_type: Type::Unit,
            value_kind: ValueKind::RValue,
            kind: EExprKind::Assign(Box::new(a), Box::new(b)),
        }
    }
    fn check_add(&self, a: EExpr, b: EExpr) -> EExpr {
        use Type::*;
        match (a.expr_type, b.expr_type) {
            (Literal(a), Literal(b)) => {
                let value = a.wrapping_add(b);
                EExpr {
                    expr_type: Type::Literal(value),
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::Decimal(value),
                }
            }
            (Integer(_), Literal(_)) => self.check_int_literal_add(a, b),
            (Literal(_), Integer(_)) => self.check_int_literal_add(b, a),
            (Integer(_), Integer(_)) => self.check_integer_add(a, b),
            _ => panic!(),
        }
    }
    fn check_int_literal_add(&self, i: EExpr, l: EExpr) -> EExpr {
        let Type::Integer(it) = i.expr_type else { panic!() };
        let Type::Literal(v) = l.expr_type else { panic!() };

        assert!(it.fits_value(v));
        let l = self.coerce(l, it);
        self.check_add(i, l)
    }
    fn check_integer_add(&self, a: EExpr, b: EExpr) -> EExpr {
        let Type::Integer(at) = a.expr_type else { panic!() };
        let Type::Integer(bt) = b.expr_type else { panic!() };

        use IntType::*;
        match (at, bt) {
            (Regular(at, ats), Regular(bt, bts)) => {
                let higher_rank = at.max(bt);
                let any_unsigned = !ats || !bts;
                let a = self.coerce(a, IntType::Regular(higher_rank, ats));
                let b = self.coerce(b, IntType::Regular(higher_rank, bts));

                EExpr {
                    expr_type: IntType::Regular(higher_rank, !any_unsigned).into(),
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::Add(Box::new(a), Box::new(b)),
                }
            }
            (Special(at, ats), Special(bt, bts)) => {
                assert_eq!(at, bt);
                let any_unsigned = !ats || !bts;

                EExpr {
                    expr_type: IntType::Special(at, !any_unsigned).into(),
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::Add(Box::new(a), Box::new(b)),
                }
            }
            _ => panic!(),
        }
    }
    fn check_sub(&self, a: EExpr, b: EExpr) -> EExpr {
        use Type::*;
        match (a.expr_type, b.expr_type) {
            (Literal(a), Literal(b)) => {
                let value = a.wrapping_sub(b);
                EExpr {
                    expr_type: Type::Literal(value),
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::Decimal(value),
                }
            }
            (Integer(_), Literal(_)) => self.check_int_literal_sub(a, b),
            (Literal(_), Integer(_)) => self.check_int_literal_sub(b, a),
            (Integer(_), Integer(_)) => self.check_integer_sub(a, b),
            _ => panic!(),
        }
    }
    fn check_int_literal_sub(&self, i: EExpr, l: EExpr) -> EExpr {
        let Type::Integer(it) = i.expr_type else { panic!() };
        let Type::Literal(v) = l.expr_type else { panic!() };

        assert!(it.fits_value(v));
        let l = self.coerce(l, it);
        self.check_sub(i, l)
    }
    fn check_integer_sub(&self, a: EExpr, b: EExpr) -> EExpr {
        let Type::Integer(at) = a.expr_type else { panic!() };
        let Type::Integer(bt) = b.expr_type else { panic!() };

        use IntType::*;
        match (at, bt) {
            (Regular(at, ats), Regular(bt, bts)) => {
                let higher_rank = at.max(bt);
                let a = self.coerce(a, IntType::Regular(higher_rank, ats));
                let b = self.coerce(b, IntType::Regular(higher_rank, bts));
                assert!(!ats || bts);

                EExpr {
                    expr_type: IntType::Regular(higher_rank, ats).into(),
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::Sub(Box::new(a), Box::new(b)),
                }
            }
            (Special(at, ats), Special(bt, bts)) => {
                assert_eq!(at, bt);
                assert!(!ats || bts);

                EExpr {
                    expr_type: IntType::Special(at, ats).into(),
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::Sub(Box::new(a), Box::new(b)),
                }
            }
            _ => panic!(),
        }
    }
    fn check_mul(&self, a: EExpr, b: EExpr) -> EExpr {
        use Type::*;
        match (a.expr_type, b.expr_type) {
            (Literal(a), Literal(b)) => {
                let value = a.wrapping_mul(b);
                EExpr {
                    expr_type: Literal(value),
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::Decimal(value),
                }
            }
            (Literal(_), Integer(bt)) => {
                let a = self.coerce(a, bt);
                self.check_mul(a, b)
            }
            (Integer(at), Literal(_)) => {
                let b = self.coerce(b, at);
                self.check_mul(a, b)
            }
            (Integer(at), Integer(bt)) => {
                let common = Self::common_int_type(at, bt);
                let a = self.coerce(a, common);
                let b = self.coerce(b, common);

                EExpr {
                    expr_type: common.into(),
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::Mul(a.into(), b.into()),
                }
            }
            _ => panic!(),
        }
    }
    fn check_div(&self, a: EExpr, b: EExpr) -> EExpr {
        use Type::*;
        match (a.expr_type, b.expr_type) {
            (Literal(a), Literal(b)) => {
                let value = a.wrapping_div(b);
                EExpr {
                    expr_type: Literal(value),
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::Decimal(value),
                }
            }
            (Literal(_), Integer(bt)) => {
                let a = self.coerce(a, bt);
                self.check_div(a, b)
            }
            (Integer(at), Literal(_)) => {
                let b = self.coerce(b, at);
                self.check_div(a, b)
            }
            (Integer(at), Integer(bt)) => {
                let common = Self::common_int_type(at, bt);
                let a = self.coerce(a, common);
                let b = self.coerce(b, common);

                EExpr {
                    expr_type: common.into(),
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::Div(a.into(), b.into()),
                }
            }
            _ => panic!(),
        }
    }
    fn check_shift_left(&self, a: EExpr, b: EExpr) -> EExpr {
        use Type::*;
        match (a.expr_type, b.expr_type) {
            (Literal(a), Literal(b)) => {
                let value = a.wrapping_shl(b.try_into().unwrap());
                EExpr {
                    expr_type: Literal(value),
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::Decimal(value),
                }
            }
            (Literal(_), Integer(_)) => {
                let a = self.make_literal_concrete(a);
                self.check_shift_left(a, b)
            }
            (Integer(_), Literal(_)) => {
                let b = self.coerce(b, IntType::usize());
                self.check_shift_left(a, b)
            }
            (Integer(at), Integer(bt)) => {
                assert!(!bt.signed());

                EExpr {
                    expr_type: at.into(),
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::ShiftLeft(a.into(), b.into()),
                }
            }
            _ => panic!(),
        }
    }
    fn check_shift_right(&self, a: EExpr, b: EExpr) -> EExpr {
        use Type::*;
        match (a.expr_type, b.expr_type) {
            (Literal(a), Literal(b)) => {
                let value = a.wrapping_shr(b.try_into().unwrap());
                EExpr {
                    expr_type: Literal(value),
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::Decimal(value),
                }
            }
            (Literal(_), Integer(_)) => {
                let a = self.make_literal_concrete(a);
                self.check_shift_right(a, b)
            }
            (Integer(_), Literal(_)) => {
                let b = self.coerce(b, IntType::usize());
                self.check_shift_right(a, b)
            }
            (Integer(at), Integer(bt)) => {
                assert!(!bt.signed());

                EExpr {
                    expr_type: at.into(),
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::ShiftRight(a.into(), b.into()),
                }
            }
            _ => panic!(),
        }
    }
    fn check_test_equal(&self, a: EExpr, b: EExpr) -> EExpr {
        use Type::*;
        match (a.expr_type, b.expr_type) {
            (Bool, Bool) => EExpr {
                expr_type: Type::Bool,
                value_kind: ValueKind::RValue,
                kind: EExprKind::TestEqual(a.into(), b.into()),
            },
            (Literal(a), Literal(b)) => {
                let value = a == b;
                EExpr {
                    expr_type: Bool,
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::Boolean(value),
                }
            }
            (Literal(_), Integer(bt)) => {
                let a = self.coerce(a, bt);
                self.check_test_equal(a, b)
            }
            (Integer(at), Literal(_)) => {
                let b = self.coerce(b, at);
                self.check_test_equal(a, b)
            }
            (Integer(at), Integer(bt)) => {
                let common = Self::common_int_type(at, bt);
                let a = self.coerce(a, common);
                let b = self.coerce(b, common);

                EExpr {
                    expr_type: Bool,
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::TestEqual(a.into(), b.into()),
                }
            }
            _ => panic!(),
        }
    }
    fn check_test_not_equal(&self, a: EExpr, b: EExpr) -> EExpr {
        use Type::*;
        match (a.expr_type, b.expr_type) {
            (Bool, Bool) => EExpr {
                expr_type: Type::Bool,
                value_kind: ValueKind::RValue,
                kind: EExprKind::TestNotEqual(a.into(), b.into()),
            },
            (Literal(a), Literal(b)) => {
                let value = a != b;
                EExpr {
                    expr_type: Bool,
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::Boolean(value),
                }
            }
            (Literal(_), Integer(bt)) => {
                let a = self.coerce(a, bt);
                self.check_test_not_equal(a, b)
            }
            (Integer(at), Literal(_)) => {
                let b = self.coerce(b, at);
                self.check_test_not_equal(a, b)
            }
            (Integer(at), Integer(bt)) => {
                let common = Self::common_int_type(at, bt);
                let a = self.coerce(a, common);
                let b = self.coerce(b, common);

                EExpr {
                    expr_type: Bool,
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::TestNotEqual(a.into(), b.into()),
                }
            }
            _ => panic!(),
        }
    }
    fn check_test_greater(&self, a: EExpr, b: EExpr) -> EExpr {
        use Type::*;
        match (a.expr_type, b.expr_type) {
            (Literal(a), Literal(b)) => {
                let value = a > b;
                EExpr {
                    expr_type: Bool,
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::Boolean(value),
                }
            }
            (Literal(_), Integer(bt)) => {
                let a = self.coerce(a, bt);
                self.check_test_greater(a, b)
            }
            (Integer(at), Literal(_)) => {
                let b = self.coerce(b, at);
                self.check_test_greater(a, b)
            }
            (Integer(at), Integer(bt)) => {
                let common = Self::common_int_type(at, bt);
                let a = self.coerce(a, common);
                let b = self.coerce(b, common);

                EExpr {
                    expr_type: Bool,
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::TestGreater(a.into(), b.into()),
                }
            }
            _ => panic!(),
        }
    }
    fn check_test_greater_equal(&self, a: EExpr, b: EExpr) -> EExpr {
        use Type::*;
        match (a.expr_type, b.expr_type) {
            (Literal(a), Literal(b)) => {
                let value = a >= b;
                EExpr {
                    expr_type: Bool,
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::Boolean(value),
                }
            }
            (Literal(_), Integer(bt)) => {
                let a = self.coerce(a, bt);
                self.check_test_greater_equal(a, b)
            }
            (Integer(at), Literal(_)) => {
                let b = self.coerce(b, at);
                self.check_test_greater_equal(a, b)
            }
            (Integer(at), Integer(bt)) => {
                let common = Self::common_int_type(at, bt);
                let a = self.coerce(a, common);
                let b = self.coerce(b, common);

                EExpr {
                    expr_type: Bool,
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::TestGreaterEqual(a.into(), b.into()),
                }
            }
            _ => panic!(),
        }
    }
    fn check_test_less(&self, a: EExpr, b: EExpr) -> EExpr {
        use Type::*;
        match (a.expr_type, b.expr_type) {
            (Literal(a), Literal(b)) => {
                let value = a < b;
                EExpr {
                    expr_type: Bool,
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::Boolean(value),
                }
            }
            (Literal(_), Integer(bt)) => {
                let a = self.coerce(a, bt);
                self.check_test_less(a, b)
            }
            (Integer(at), Literal(_)) => {
                let b = self.coerce(b, at);
                self.check_test_less(a, b)
            }
            (Integer(at), Integer(bt)) => {
                let common = Self::common_int_type(at, bt);
                let a = self.coerce(a, common);
                let b = self.coerce(b, common);

                EExpr {
                    expr_type: Bool,
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::TestLess(a.into(), b.into()),
                }
            }
            _ => panic!(),
        }
    }
    fn check_test_less_equal(&self, a: EExpr, b: EExpr) -> EExpr {
        use Type::*;
        match (a.expr_type, b.expr_type) {
            (Literal(a), Literal(b)) => {
                let value = a <= b;
                EExpr {
                    expr_type: Bool,
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::Boolean(value),
                }
            }
            (Literal(_), Integer(bt)) => {
                let a = self.coerce(a, bt);
                self.check_test_less_equal(a, b)
            }
            (Integer(at), Literal(_)) => {
                let b = self.coerce(b, at);
                self.check_test_less_equal(a, b)
            }
            (Integer(at), Integer(bt)) => {
                let common = Self::common_int_type(at, bt);
                let a = self.coerce(a, common);
                let b = self.coerce(b, common);

                EExpr {
                    expr_type: Bool,
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::TestLessEqual(a.into(), b.into()),
                }
            }
            _ => panic!(),
        }
    }
    fn check_bitand(&self, a: EExpr, b: EExpr) -> EExpr {
        use Type::*;
        match (a.expr_type, b.expr_type) {
            (Literal(av), Literal(bv)) => {
                if av.unsigned_abs() > bv.unsigned_abs() {
                    let a = self.make_literal_concrete(a);
                    self.check_bitand(a, b)
                } else {
                    let b = self.make_literal_concrete(b);
                    self.check_bitand(a, b)
                }
            }
            (Literal(_), Integer(bt)) => {
                let a = self.coerce(a, bt);
                self.check_bitand(a, b)
            }
            (Integer(at), Literal(_)) => {
                let b = self.coerce(b, at);
                self.check_bitand(a, b)
            }
            (Integer(at), Integer(bt)) => {
                let common = Self::common_int_type(at, bt);
                let a = self.coerce(a, common);
                let b = self.coerce(b, common);

                EExpr {
                    expr_type: common.into(),
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::BitAnd(a.into(), b.into()),
                }
            }
            _ => panic!(),
        }
    }
    fn check_bitor(&self, a: EExpr, b: EExpr) -> EExpr {
        use Type::*;
        match (a.expr_type, b.expr_type) {
            (Literal(av), Literal(bv)) => {
                if av.unsigned_abs() > bv.unsigned_abs() {
                    let a = self.make_literal_concrete(a);
                    self.check_bitor(a, b)
                } else {
                    let b = self.make_literal_concrete(b);
                    self.check_bitor(a, b)
                }
            }
            (Literal(_), Integer(bt)) => {
                let a = self.coerce(a, bt);
                self.check_bitor(a, b)
            }
            (Integer(at), Literal(_)) => {
                let b = self.coerce(b, at);
                self.check_bitor(a, b)
            }
            (Integer(at), Integer(bt)) => {
                let common = Self::common_int_type(at, bt);
                let a = self.coerce(a, common);
                let b = self.coerce(b, common);

                EExpr {
                    expr_type: common.into(),
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::BitOr(a.into(), b.into()),
                }
            }
            _ => panic!(),
        }
    }
    fn check_bitxor(&self, a: EExpr, b: EExpr) -> EExpr {
        use Type::*;
        match (a.expr_type, b.expr_type) {
            (Literal(av), Literal(bv)) => {
                if av.unsigned_abs() > bv.unsigned_abs() {
                    let a = self.make_literal_concrete(a);
                    self.check_bitxor(a, b)
                } else {
                    let b = self.make_literal_concrete(b);
                    self.check_bitxor(a, b)
                }
            }
            (Literal(_), Integer(bt)) => {
                let a = self.coerce(a, bt);
                self.check_bitxor(a, b)
            }
            (Integer(at), Literal(_)) => {
                let b = self.coerce(b, at);
                self.check_bitxor(a, b)
            }
            (Integer(at), Integer(bt)) => {
                let common = Self::common_int_type(at, bt);
                let a = self.coerce(a, common);
                let b = self.coerce(b, common);

                EExpr {
                    expr_type: common.into(),
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::BitXor(a.into(), b.into()),
                }
            }
            _ => panic!(),
        }
    }
    fn check_and(&self, a: EExpr, b: EExpr) -> EExpr {
        let a = self.coerce(a, Type::Bool);
        let b = self.coerce(b, Type::Bool);

        EExpr {
            expr_type: Type::Bool,
            value_kind: ValueKind::RValue,
            kind: EExprKind::And(a.into(), b.into()),
        }
    }
    fn check_or(&self, a: EExpr, b: EExpr) -> EExpr {
        let a = self.coerce(a, Type::Bool);
        let b = self.coerce(b, Type::Bool);

        EExpr {
            expr_type: Type::Bool,
            value_kind: ValueKind::RValue,
            kind: EExprKind::Or(a.into(), b.into()),
        }
    }
    fn common_int_type(a: IntType, b: IntType) -> IntType {
        use IntType::*;
        match (a, b) {
            (Regular(at, ats), Regular(bt, bts)) => {
                let higher = at.max(bt);
                IntType::Regular(higher, ats | bts)
            }
            (Special(at, ats), Special(bt, bts)) => {
                assert_eq!(at, bt);
                IntType::Special(at, ats | bts)
            }
            _ => panic!(),
        }
    }
    fn check_prefix_op(&mut self, op: PrefixOp, a: &Expr<'a>) -> EExpr {
        let a = self.check_expr(a);

        use PrefixOp::*;
        match op {
            BitNot => self.check_bitnot(a),
            Not => self.check_not(a),
            Negate => self.check_negate(a),
            AddrOf => self.check_addr_of(a),
            AddrOfMut => self.check_addr_of_mut(a),
            Deref => self.check_deref(a),
        }
    }
    fn check_bitnot(&self, a: EExpr) -> EExpr {
        let a = self.make_literal_concrete(a);
        assert!(a.expr_type.is_integer());

        EExpr {
            expr_type: a.expr_type,
            value_kind: ValueKind::RValue,
            kind: EExprKind::BitNot(Box::new(a)),
        }
    }
    fn check_not(&self, a: EExpr) -> EExpr {
        let a = self.coerce(a, Type::Bool);
        EExpr {
            expr_type: Type::Bool,
            value_kind: ValueKind::RValue,
            kind: EExprKind::Not(Box::new(a)),
        }
    }
    fn check_negate(&self, a: EExpr) -> EExpr {
        if let Type::Literal(value) = a.expr_type {
            let value = value.wrapping_neg();
            return EExpr {
                expr_type: Type::Literal(value),
                value_kind: ValueKind::RValue,
                kind: EExprKind::Decimal(value),
            };
        }

        assert!(a.expr_type.is_integer());
        EExpr {
            expr_type: a.expr_type,
            value_kind: ValueKind::RValue,
            kind: EExprKind::Negate(Box::new(a)),
        }
    }
    fn check_addr_of(&mut self, a: EExpr) -> EExpr {
        assert!(matches!(a.value_kind, ValueKind::LValue(_)));
        let ptr_t = self.types.add_pointer_type(false, a.expr_type);
        EExpr {
            expr_type: ptr_t.into(),
            value_kind: ValueKind::RValue,
            kind: EExprKind::AddrOf(Box::new(a)),
        }
    }
    fn check_addr_of_mut(&mut self, a: EExpr) -> EExpr {
        assert!(matches!(a.value_kind, ValueKind::LValue(true)));
        let ptr_t = self.types.add_pointer_type(true, a.expr_type);
        EExpr {
            expr_type: ptr_t.into(),
            value_kind: ValueKind::RValue,
            kind: EExprKind::AddrOfMut(Box::new(a)),
        }
    }
    fn check_deref(&self, a: EExpr) -> EExpr {
        let Type::Pointer(id) = a.expr_type else { panic!() };
        let mutable = self.types[id].mutable();
        let pointee = self.types[id].pointee();

        EExpr {
            expr_type: pointee,
            value_kind: ValueKind::LValue(mutable),
            kind: EExprKind::Deref(Box::new(a)),
        }
    }
    fn check_tuple(&mut self, members: &[Expr<'a>]) -> EExpr {
        if members.is_empty() {
            return EExpr::unit();
        }

        let members: Vec<_> = members
            .iter()
            .map(|m| {
                let m = self.check_expr(m);
                let m = self.make_literal_concrete(m);
                assert!(m.expr_type.is_sized());
                m
            })
            .collect();

        let types = members.iter().map(|m| m.expr_type).collect();
        let tuple_t = self.types.add_tuple_type(types).into();

        EExpr {
            expr_type: tuple_t,
            value_kind: ValueKind::RValue,
            kind: EExprKind::Tuple(members),
        }
    }
    fn check_short_array(&mut self, init: &Expr<'a>, length: &Expr<'a>) -> EExpr {
        let length = self.eval_const(length).try_into().unwrap();
        let init = self.check_expr(init);
        let init = self.make_literal_concrete(init);
        let arr_type = self.types.add_array_type(init.expr_type, length);

        EExpr {
            expr_type: arr_type.into(),
            value_kind: ValueKind::RValue,
            kind: EExprKind::ShortArray(Box::new(init), length),
        }
    }
    fn check_array(&mut self, values: &[Expr<'a>]) -> EExpr {
        assert!(!values.is_empty());
        let first_member = self.check_expr(&values[0]);
        let first_member = self.make_literal_concrete(first_member);
        let member_type = first_member.expr_type;
        let mut members = vec![first_member];
        for value in &values[1..] {
            let member = self.check_expr(value);
            let member = self.coerce(member, member_type);
            members.push(member);
        }

        let length = values.len() as u64;
        let arr_type = self.types.add_array_type(member_type, length);
        EExpr {
            expr_type: arr_type.into(),
            value_kind: ValueKind::RValue,
            kind: EExprKind::Array(members),
        }
    }
    fn check_named(&mut self, name: &str) -> EExpr {
        let Some(resolution) = self.resolve(name) else { panic!("Could not resolve identifier {name}") };

        match resolution {
            Resolution::Variable(id) => {
                let mutable = self.symbols[id].mutable();
                let expr_type = self.symbols[id].var_type();
                EExpr {
                    expr_type,
                    value_kind: ValueKind::LValue(mutable),
                    kind: EExprKind::Variable(id),
                }
            }
            Resolution::Function(id) => {
                let ret_type = self.symbols[id].return_type();
                let param_types = self.symbols[id]
                    .parameters()
                    .iter()
                    .map(|&p| self.symbols[p].var_type())
                    .collect();
                let expr_type = self.types.add_func_type(ret_type, param_types).into();
                EExpr {
                    expr_type,
                    value_kind: ValueKind::RValue,
                    kind: EExprKind::Function(id),
                }
            }
        }
    }
    fn check_decimal(&self, value: i128, suffix: Option<IntSuffix>) -> EExpr {
        let int_type = suffix
            .map(IntSuffix::to_type)
            .map(Into::into)
            .unwrap_or(Type::Literal(value));
        EExpr {
            expr_type: int_type,
            value_kind: ValueKind::RValue,
            kind: EExprKind::Decimal(value),
        }
    }
    fn check_bool(&self, value: bool) -> EExpr {
        EExpr {
            expr_type: Type::Bool,
            value_kind: ValueKind::RValue,
            kind: EExprKind::Boolean(value),
        }
    }

    fn make_literal_concrete(&self, e: EExpr) -> EExpr {
        if let Type::Literal(value) = e.expr_type {
            self.coerce(e, Self::best_literal_fit(value))
        } else {
            e
        }
    }
    fn best_literal_fit(value: i128) -> IntType {
        if IntType::i32().fits_value(value) {
            IntType::i32()
        } else if IntType::i64().fits_value(value) {
            IntType::i64()
        } else if IntType::u64().fits_value(value) {
            IntType::u64()
        } else {
            panic!()
        }
    }

    fn coerce(&self, e: EExpr, into: impl Into<Type>) -> EExpr {
        let into = into.into();

        if e.expr_type == into {
            return e;
        }

        assert!(
            self.can_coerce(e.expr_type, into),
            "Cannot coerce {:?} into {into:?}",
            e.expr_type
        );

        EExpr {
            expr_type: into,
            value_kind: ValueKind::RValue,
            kind: EExprKind::Conversion(Box::new(e)),
        }
    }
    fn can_coerce(&self, from: Type, to: Type) -> bool {
        use Type::*;
        match (from, to) {
            (a, b) if a == b => true,
            (Literal(value), to) => Self::can_coerce_literal(value, to),
            (Integer(a), Integer(b)) => Self::can_coerce_int(a, b),
            (Pointer(p), to) => self.can_coerce_pointer(p, to),
            _ => false,
        }
    }
    fn can_coerce_literal(value: i128, to: Type) -> bool {
        let Type::Integer(to) = to else { return false };
        to.fits_value(value)
    }
    fn can_coerce_int(from: IntType, to: IntType) -> bool {
        use IntType::*;
        match (from, to) {
            (a, b) if a == b => true,
            (Regular(f, fs), Regular(t, ts)) => (fs == ts || ts) && f < t,
            (Regular(_, _), Special(_, s)) => {
                Self::can_coerce_int(from, IntType::Regular(RegularIntKind::Short, s))
            }
            (Special(_, false), Regular(RegularIntKind::Long, false)) => true,
            _ => false,
        }
    }
    fn can_coerce_pointer(&self, from: PointerTypeID, to: Type) -> bool {
        let Type::Pointer(to) = to else { return false };
        let from_mutable = self.types[from].mutable();
        let to_mutable = self.types[to].mutable();
        assert!(from_mutable || !to_mutable);

        let from_pointee = self.types[from].pointee();
        let to_pointee = self.types[to].pointee();
        if from_pointee == to_pointee {
            return true;
        }

        if to_pointee.is_void() {
            return true;
        }

        use Type::*;
        match (from_pointee, to_pointee) {
            (Array(rid), Slice(sid)) => {
                let arr_member = self.types[rid].member();
                let sli_member = self.types[sid].member();
                arr_member == sli_member || sli_member.is_void()
            }
            (Slice(a), Slice(b)) => {
                let a_member = self.types[a].member();
                let b_member = self.types[b].member();
                a_member == b_member || a_member.is_void() || b_member.is_void()
            }
            _ => false,
        }
    }

    fn check_type_expr(&mut self, e: &TypeExpr) -> Type {
        match &e.kind {
            &TypeExprKind::Named(name) => match name {
                "void" => Type::Void,
                "bool" => Type::Bool,
                "u8" => IntType::u8().into(),
                "i8" => IntType::i8().into(),
                "u16" => IntType::u16().into(),
                "i16" => IntType::i16().into(),
                "u32" => IntType::u32().into(),
                "i32" => IntType::i32().into(),
                "u64" => IntType::u64().into(),
                "i64" => IntType::i64().into(),
                "usize" => IntType::usize().into(),
                "isize" => IntType::isize().into(),
                "uptr" => IntType::uptr().into(),
                "iptr" => IntType::iptr().into(),
                _ => panic!(),
            },
            &TypeExprKind::Pointer(mutable, ref pointee) => {
                let pointee = self.check_type_expr(pointee);
                assert!(pointee.is_concrete() || pointee.is_void());
                self.types.add_pointer_type(mutable, pointee).into()
            }
            &TypeExprKind::Array(ref len, ref member) => {
                let member = self.check_type_expr(member);
                assert!(member.is_sized());
                let length = self.eval_const(len).try_into().unwrap();
                self.types.add_array_type(member, length).into()
            }
            TypeExprKind::Tuple(members) => {
                let members: Vec<_> = members
                    .iter()
                    .map(|m| {
                        let mem_t = self.check_type_expr(m);
                        assert!(mem_t.is_sized());
                        mem_t
                    })
                    .collect();

                if members.is_empty() {
                    Type::Unit
                } else {
                    self.types.add_tuple_type(members).into()
                }
            }
            &TypeExprKind::Slice(ref member) => {
                let member = self.check_type_expr(member);
                assert!(member.is_sized() || member.is_void());
                self.types.add_slice_type(member).into()
            }
        }
    }
    fn eval_const(&self, e: &Expr) -> i128 {
        match &e.kind {
            &ExprKind::Decimal(value, _) => value,
            _ => panic!(),
        }
    }

    fn enter_scope(&mut self) {
        self.scopes.push(Scope::new());
    }
    fn leave_scope(&mut self) {
        self.scopes.pop();
    }
    fn register(&mut self, name: &'a str, id: VarID) {
        self.scopes.last_mut().unwrap().register(name, id);
    }
    fn resolve(&self, name: &str) -> Option<Resolution> {
        for scope in self.scopes.iter().rev() {
            if let Some(res) = scope.resolve(name) {
                return Some(res);
            }
        }

        self.symbols
            .resolve_function(name)
            .map(|f| Resolution::Function(f))
    }
}

struct Scope<'a> {
    vars: HashMap<&'a str, VarID>,
}
impl<'a> Scope<'a> {
    fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }

    fn register(&mut self, name: &'a str, id: VarID) {
        self.vars.insert(name, id);
    }

    fn resolve(&self, name: &str) -> Option<Resolution> {
        self.vars.get(name).map(|&v| Resolution::Variable(v))
    }
}

enum Resolution {
    Function(FunctionID),
    Variable(VarID),
}
