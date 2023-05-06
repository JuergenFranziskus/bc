use crate::frontend::ast::Intrinsic;

use super::{
    ast::{
        Ast, BinaryOp, Expr, ExprKind, Function, FunctionParameter, IntSuffix, PrefixOp, TypeExpr,
        TypeExprKind,
    },
    lexer::{Token, TokenKind},
    span::Span,
};

pub struct Parser<'a, 'b> {
    tokens: &'b [Token<'a>],
    index: usize,
}
impl<'a, 'b> Parser<'a, 'b> {
    pub fn new(tokens: &'b [Token<'a>]) -> Self {
        Self { tokens, index: 0 }
    }

    pub fn parse(mut self) -> Ast<'a> {
        let mut functions = Vec::new();
        while !self.at_end() {
            functions.push(self.parse_function());
        }

        Ast { functions }
    }

    fn parse_function(&mut self) -> Function<'a> {
        let (name, start) = self.parse_identifier();
        self.consume(TokenKind::DoubleColon);
        let parameters = self.parse_function_parameters();
        let mut return_type = None;
        if !self.is(TokenKind::DoubleRightArrow) {
            return_type = Some(self.parse_type_expr());
        }
        self.consume(TokenKind::DoubleRightArrow);

        let body = self.parse_expr();

        if !body.is_structural() || self.is(TokenKind::Semicolon) {
            self.consume(TokenKind::Semicolon);
        }

        let span = Span::merge(start, body.span);
        Function {
            span,
            name,
            return_type,
            parameters,
            body,
        }
    }
    fn parse_function_parameters(&mut self) -> Vec<FunctionParameter<'a>> {
        self.consume(TokenKind::OpenParen);
        let mut params = Vec::new();

        while !self.is(TokenKind::CloseParen) {
            params.push(self.parse_function_parameter());
            if self.is(TokenKind::Comma) {
                self.next()
            } else {
                break;
            }
        }
        self.consume(TokenKind::CloseParen);

        params
    }
    fn parse_function_parameter(&mut self) -> FunctionParameter<'a> {
        let mut mutable = false;
        if self.is(TokenKind::MutKeyword) {
            mutable = true;
            self.next();
        }

        let (name, start) = self.parse_identifier();
        if self.is(TokenKind::Colon) {
            self.next()
        }
        let param_type = self.parse_type_expr();
        let span = Span::merge(start, param_type.span);

        FunctionParameter {
            span,
            name,
            param_type,
            mutable,
        }
    }

    fn parse_expr(&mut self) -> Expr<'a> {
        self.parse_expr_bp(i16::MIN)
    }
    fn parse_expr_bp(&mut self, min_bp: i16) -> Expr<'a> {
        let mut lhs = self.parse_cast_expr();

        while !self.at_end() {
            let op = match self.curr().kind {
                TokenKind::Equal => BinaryOp::Assign,
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Sub,
                TokenKind::Star => BinaryOp::Mul,
                TokenKind::Slash => BinaryOp::Div,
                TokenKind::DoubleLess => BinaryOp::ShiftLeft,
                TokenKind::DoubleGreater => BinaryOp::ShiftRight,
                TokenKind::DoubleEqual => BinaryOp::TestEqual,
                TokenKind::GreaterThan => BinaryOp::TestGreater,
                TokenKind::LessThan => BinaryOp::TestLess,
                TokenKind::GreaterEqual => BinaryOp::TestGreaterEqual,
                TokenKind::LessEqual => BinaryOp::TestLessEqual,
                TokenKind::Ampersand => BinaryOp::BitAnd,
                TokenKind::DoubleAmpersand => BinaryOp::And,
                TokenKind::Pipe => BinaryOp::BitOr,
                TokenKind::DoublePipe => BinaryOp::Or,
                TokenKind::Caret => BinaryOp::BitXor,
                _ => break,
            };

            let (l_bp, r_bp) = infix_binding_power(op);
            if l_bp < min_bp {
                break;
            }
            self.next();

            let rhs = self.parse_expr_bp(r_bp);
            let span = Span::merge(lhs.span, rhs.span);
            lhs = Expr {
                span,
                kind: ExprKind::Binary(op, lhs.into(), rhs.into()),
            }
        }

        lhs
    }

    fn parse_cast_expr(&mut self) -> Expr<'a> {
        let mut lhs = self.parse_prefix_expr();

        while !self.at_end() {
            match self.curr().kind {
                TokenKind::AsKeyword => {
                    self.next();
                    let to_type = self.parse_type_expr();
                    lhs = Expr {
                        span: Span::merge(lhs.span, to_type.span),
                        kind: ExprKind::BitCast(Box::new(lhs), Box::new(to_type)),
                    };
                }
                TokenKind::CastKeyword => {
                    self.next();
                    let to_type = self.parse_type_expr();
                    lhs = Expr {
                        span: Span::merge(lhs.span, to_type.span),
                        kind: ExprKind::Cast(Box::new(lhs), Box::new(to_type)),
                    };
                }
                _ => break,
            }
        }

        lhs
    }
    fn parse_prefix_expr(&mut self) -> Expr<'a> {
        let start = self.curr_span();
        let op = match self.curr().kind {
            TokenKind::Minus => PrefixOp::Negate,
            TokenKind::Tilde => PrefixOp::BitNot,
            TokenKind::Exclamation => PrefixOp::Not,
            TokenKind::Ampersand => {
                let mutable = self.peek_is(TokenKind::MutKeyword);
                if mutable {
                    self.next();
                    PrefixOp::AddrOfMut
                } else {
                    PrefixOp::AddrOf
                }
            }
            TokenKind::Star => PrefixOp::Deref,
            _ => return self.parse_postfix_expr(),
        };
        self.next();
        let rhs = self.parse_prefix_expr();

        Expr {
            span: Span::merge(start, rhs.span),
            kind: ExprKind::Prefix(op, Box::new(rhs)),
        }
    }
    fn parse_postfix_expr(&mut self) -> Expr<'a> {
        let mut lhs = self.parse_leaf_expr();

        while !self.at_end() {
            match self.curr().kind {
                TokenKind::Dot => lhs = self.parse_member_expr(lhs),
                TokenKind::OpenParen => lhs = self.parse_call_expr(lhs),
                TokenKind::OpenSquare => lhs = self.parse_index_expr(lhs),
                _ => break,
            }
        }

        lhs
    }

    fn parse_member_expr(&mut self, lhs: Expr<'a>) -> Expr<'a> {
        self.consume(TokenKind::Dot);
        let end = self.curr_span();
        let TokenKind::Decimal(src) = self.curr().kind else { panic!() };
        let index = src.parse().unwrap();
        self.next();

        Expr {
            span: Span::merge(lhs.span, end),
            kind: ExprKind::TupleIndex(lhs.into(), index),
        }
    }
    fn parse_call_expr(&mut self, lhs: Expr<'a>) -> Expr<'a> {
        self.consume(TokenKind::OpenParen);
        let mut args = Vec::new();

        while !self.is(TokenKind::CloseParen) {
            args.push(self.parse_expr());

            if self.is(TokenKind::Comma) {
                self.next();
            } else {
                break;
            }
        }
        let end = self.consume(TokenKind::CloseParen);

        let span = Span::merge(lhs.span, end);
        Expr {
            span,
            kind: ExprKind::Call(Box::new(lhs), args),
        }
    }
    fn parse_index_expr(&mut self, lhs: Expr<'a>) -> Expr<'a> {
        self.consume(TokenKind::OpenSquare);
        let index = self.parse_expr();
        let end = self.consume(TokenKind::CloseSquare);

        Expr {
            span: Span::merge(lhs.span, end),
            kind: ExprKind::Index(Box::new(lhs), Box::new(index)),
        }
    }
    fn parse_leaf_expr(&mut self) -> Expr<'a> {
        match self.curr().kind {
            TokenKind::At => self.parse_intrinsic_expr(),
            TokenKind::MakeSliceKeyword => self.parse_make_slice(),
            TokenKind::SLenKeyword => self.parse_slice_len(),
            TokenKind::OpenParen => self.parse_paren_expr(),
            TokenKind::OpenSquare => self.parse_array_expr(),
            TokenKind::Decimal(_) => self.parse_decimal_expr(),
            TokenKind::OpenCurly => self.parse_block_expr(),
            TokenKind::LetKeyword => self.parse_declaration(),
            TokenKind::WhileKeyword => self.parse_while_expr(),
            TokenKind::IfKeyword => self.parse_if_expr(),
            TokenKind::Identifier(_) => self.parse_identifier_expr(),
            TokenKind::TrueKeyword | TokenKind::FalseKeyword => self.parse_boolean_expr(),
            _ => panic!("Expected leaf expression, found {:?}", self.curr()),
        }
    }
    fn parse_intrinsic_expr(&mut self) -> Expr<'a> {
        let start = self.consume(TokenKind::At);
        let (name, _) = self.parse_identifier();
        let intrinsic = match name {
            "storevolatile" => Intrinsic::VolatileStore,
            _ => panic!(),
        };

        let mut args = Vec::new();
        self.consume(TokenKind::OpenParen);
        while !self.is(TokenKind::CloseParen) {
            args.push(self.parse_expr());
            if self.is(TokenKind::Comma) {
                self.next();
            } else {
                break;
            }
        }
        let end = self.consume(TokenKind::CloseParen);
        let span = Span::merge(start, end);

        Expr {
            span,
            kind: ExprKind::Intrinsic(intrinsic, args),
        }
    }
    fn parse_while_expr(&mut self) -> Expr<'a> {
        let start = self.consume(TokenKind::WhileKeyword);

        let condition = self.parse_expr();
        let body = self.parse_block_expr();

        Expr {
            span: Span::merge(start, body.span),
            kind: ExprKind::While(Box::new(condition), Box::new(body)),
        }
    }
    fn parse_if_expr(&mut self) -> Expr<'a> {
        let start = self.consume(TokenKind::IfKeyword);
        let con = self.parse_expr();

        let then = self.parse_block_expr();
        let els = if self.is(TokenKind::ElseKeyword) {
            self.next();
            Some(if self.is(TokenKind::IfKeyword) {
                self.parse_if_expr()
            } else {
                self.parse_block_expr()
            })
        } else {
            None
        };

        let end = els.as_ref().map(|e| e.span).unwrap_or(then.span);

        Expr {
            span: Span::merge(start, end),
            kind: ExprKind::If(Box::new(con), Box::new(then), els.map(Box::new)),
        }
    }
    fn parse_block_expr(&mut self) -> Expr<'a> {
        let start = self.consume(TokenKind::OpenCurly);

        let mut statements = Vec::new();
        let mut value = None;

        while !(self.at_end() || self.is(TokenKind::CloseCurly)) {
            if let Some(value) = value.take() {
                statements.push(value);
            }
            let expr = self.parse_expr();

            if self.is(TokenKind::Semicolon) {
                statements.push(expr);
                self.next();
            } else {
                let structural = expr.is_structural();
                value = Some(expr);
                if !structural {
                    break;
                }
            }
        }

        let end = self.consume(TokenKind::CloseCurly);

        Expr {
            span: Span::merge(start, end),
            kind: ExprKind::Block(statements, value.map(Box::new)),
        }
    }
    fn parse_decimal_expr(&mut self) -> Expr<'a> {
        let span = self.curr_span();
        let TokenKind::Decimal(src) = self.curr().kind else { panic!() };
        self.next();

        let suffix = IntSuffix::parse_from_src(src);
        let suffix_len = suffix.map(IntSuffix::src_len).unwrap_or(0);
        let number_src = &src[..src.len() - suffix_len];
        let value = number_src.parse().unwrap();

        Expr {
            span,
            kind: ExprKind::Decimal(value, suffix),
        }
    }
    fn parse_declaration(&mut self) -> Expr<'a> {
        let start = self.consume(TokenKind::LetKeyword);
        let mut mutable = false;
        if self.is(TokenKind::MutKeyword) {
            mutable = true;
            self.next();
        }
        let (name, _) = self.parse_identifier();

        let mut var_type = None;
        if self.is(TokenKind::Colon) {
            self.next();
            var_type = Some(self.parse_type_expr());
        }

        self.consume(TokenKind::Equal);
        let value = self.parse_expr();

        let span = Span::merge(start, value.span);
        Expr {
            span,
            kind: ExprKind::Declaration(name, mutable, var_type.map(Box::new), Box::new(value)),
        }
    }
    fn parse_identifier_expr(&mut self) -> Expr<'a> {
        let (name, span) = self.parse_identifier();

        Expr {
            span,
            kind: ExprKind::Named(name),
        }
    }
    fn parse_boolean_expr(&mut self) -> Expr<'a> {
        let span = self.curr_span();
        let value = match self.curr().kind {
            TokenKind::TrueKeyword => true,
            TokenKind::FalseKeyword => false,
            _ => panic!(),
        };
        self.next();
        Expr {
            span,
            kind: ExprKind::Bool(value),
        }
    }
    fn parse_paren_expr(&mut self) -> Expr<'a> {
        let start = self.consume(TokenKind::OpenParen);
        let first_value = self.parse_expr();

        if self.is(TokenKind::CloseParen) {
            let end = self.consume(TokenKind::CloseParen);
            return Expr {
                span: Span::merge(start, end),
                kind: ExprKind::Paren(Box::new(first_value)),
            };
        }

        self.consume(TokenKind::Comma);
        let mut values = vec![first_value];
        while !self.is(TokenKind::CloseParen) {
            values.push(self.parse_expr());
            if self.is(TokenKind::Comma) {
                self.next();
            } else {
                break;
            }
        }
        let end = self.consume(TokenKind::CloseParen);

        Expr {
            span: Span::merge(start, end),
            kind: ExprKind::Tuple(values),
        }
    }
    fn parse_array_expr(&mut self) -> Expr<'a> {
        let start = self.consume(TokenKind::OpenSquare);
        let first_member = self.parse_expr();

        if self.is(TokenKind::Semicolon) {
            self.next();
            let length = self.parse_expr();
            let end = self.consume(TokenKind::CloseSquare);

            Expr {
                span: Span::merge(start, end),
                kind: ExprKind::ShortArray(Box::new(first_member), Box::new(length)),
            }
        } else {
            let mut members = vec![first_member];
            self.next();
            if self.is(TokenKind::Comma) {
                self.next();
            }

            while !self.is(TokenKind::CloseSquare) {
                members.push(self.parse_expr());

                if self.is(TokenKind::Comma) {
                    self.next();
                } else {
                    break;
                }
            }
            let end = self.consume(TokenKind::CloseSquare);

            Expr {
                span: Span::merge(start, end),
                kind: ExprKind::Array(members),
            }
        }
    }
    fn parse_make_slice(&mut self) -> Expr<'a> {
        let start = self.consume(TokenKind::MakeSliceKeyword);
        self.consume(TokenKind::OpenParen);
        let ptr = self.parse_expr();
        self.consume(TokenKind::Comma);
        let len = self.parse_expr();
        let end = self.consume(TokenKind::CloseParen);

        Expr {
            span: Span::merge(start, end),
            kind: ExprKind::MakeSlice(Box::new(ptr), Box::new(len)),
        }
    }
    fn parse_slice_len(&mut self) -> Expr<'a> {
        let start = self.consume(TokenKind::SLenKeyword);
        self.consume(TokenKind::OpenParen);
        let slice = self.parse_expr();
        let end = self.consume(TokenKind::CloseParen);

        let span = Span::merge(start, end);
        Expr {
            span,
            kind: ExprKind::SliceLength(slice.into()),
        }
    }

    fn parse_type_expr(&mut self) -> TypeExpr<'a> {
        match self.curr().kind {
            TokenKind::OpenSquare => self.parse_array_type(),
            TokenKind::Identifier(_) => self.parse_named_type(),
            TokenKind::Star => self.parse_pointer_type(),
            TokenKind::OpenParen => self.parse_tuple_type(),
            _ => panic!("Expected type expression, found {:?}", self.curr()),
        }
    }
    fn parse_array_type(&mut self) -> TypeExpr<'a> {
        let start = self.consume(TokenKind::OpenSquare);
        if self.is(TokenKind::CloseSquare) {
            self.next();
            let member = self.parse_type_expr();
            return TypeExpr {
                span: Span::merge(start, member.span),
                kind: TypeExprKind::Slice(Box::new(member)),
            };
        }

        let length = self.parse_expr();
        self.consume(TokenKind::CloseSquare);
        let member = self.parse_type_expr();

        TypeExpr {
            span: Span::merge(start, member.span),
            kind: TypeExprKind::Array(length, Box::new(member)),
        }
    }
    fn parse_named_type(&mut self) -> TypeExpr<'a> {
        let (name, span) = self.parse_identifier();
        TypeExpr {
            span,
            kind: TypeExprKind::Named(name),
        }
    }
    fn parse_pointer_type(&mut self) -> TypeExpr<'a> {
        let start = self.consume(TokenKind::Star);
        let mut mutable = false;
        if self.is(TokenKind::MutKeyword) {
            mutable = true;
            self.next();
        }

        let pointee = self.parse_type_expr();

        TypeExpr {
            span: Span::merge(start, pointee.span),
            kind: TypeExprKind::Pointer(mutable, Box::new(pointee)),
        }
    }
    fn parse_tuple_type(&mut self) -> TypeExpr<'a> {
        let start = self.consume(TokenKind::OpenParen);

        let mut members = Vec::new();
        while !self.is(TokenKind::CloseParen) {
            members.push(self.parse_type_expr());

            if self.is(TokenKind::Comma) {
                self.next();
            } else {
                break;
            }
        }
        let end = self.consume(TokenKind::CloseParen);

        TypeExpr {
            span: Span::merge(start, end),
            kind: TypeExprKind::Tuple(members),
        }
    }

    fn parse_identifier(&mut self) -> (&'a str, Span) {
        let span = self.curr_span();
        let TokenKind::Identifier(name) = self.curr().kind else { panic!() };
        self.next();
        (name, span)
    }

    fn consume(&mut self, kind: TokenKind) -> Span {
        assert!(self.is(kind), "Expected {kind:?}, found {:?}", self.curr());
        let span = self.curr().span;
        self.next();
        span
    }
    fn next(&mut self) {
        self.index += 1;
    }
    fn is(&self, kind: TokenKind) -> bool {
        !self.at_end() && self.curr().kind == kind
    }
    fn peek_is(&self, kind: TokenKind) -> bool {
        self.try_peek(1).map(|t| t.kind == kind).unwrap_or(false)
    }

    fn at_end(&self) -> bool {
        self.index >= self.tokens.len()
    }
    fn try_peek(&self, offset: usize) -> Option<Token<'a>> {
        let i = self.index + offset;
        self.tokens.get(i).copied()
    }
    fn try_curr(&self) -> Option<Token<'a>> {
        self.try_peek(0)
    }
    fn curr(&self) -> Token<'a> {
        self.try_curr().unwrap()
    }
    fn curr_span(&self) -> Span {
        self.curr().span
    }
}

fn infix_binding_power(op: BinaryOp) -> (i16, i16) {
    use BinaryOp::*;
    match op {
        Assign => (101, 100),
        Or => (200, 201),
        And => (300, 301),
        TestEqual | TestNotEqual | TestGreater | TestGreaterEqual | TestLess | TestLessEqual => {
            (400, 401)
        }
        BitOr => (500, 501),
        BitXor => (600, 601),
        BitAnd => (700, 701),
        ShiftLeft | ShiftRight => (750, 751),
        Add | Sub => (800, 801),
        Mul | Div => (900, 901),
    }
}
