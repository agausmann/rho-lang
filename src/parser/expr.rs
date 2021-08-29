use crate::parser::token::Token;
use crate::parser::token::TokenKind;
use crate::value::Value;
use chumsky::prelude::*;

pub type Error = Simple<Token>;

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    RemAssign,
    AndAssign,
    OrAssign,
    BitAndAssign,
    BitOrAssign,
    BitXorAssign,
}

impl BinaryOp {
    fn precedence(&self) -> u8 {
        match self {
            Self::Assign
            | Self::AddAssign
            | Self::SubAssign
            | Self::MulAssign
            | Self::DivAssign
            | Self::RemAssign
            | Self::AndAssign
            | Self::OrAssign
            | Self::BitAndAssign
            | Self::BitOrAssign
            | Self::BitXorAssign => 0,
            Self::Or => 1,
            Self::And => 2,
            Self::Eq | Self::Ne | Self::Lt | Self::Le | Self::Gt | Self::Ge => 3,
            Self::BitOr => 4,
            Self::BitXor => 5,
            Self::BitAnd => 6,
            // TODO bit shifts
            Self::Add | Self::Sub => 8,
            Self::Mul | Self::Div | Self::Rem => 9,
        }
    }

    fn is_left_associative(&self) -> bool {
        match self {
            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::Rem
            | Self::And
            | Self::Or
            | Self::BitAnd
            | Self::BitOr
            | Self::BitXor
            | Self::Eq
            | Self::Ne
            | Self::Lt
            | Self::Le
            | Self::Gt
            | Self::Ge => true,

            Self::Assign
            | Self::AddAssign
            | Self::SubAssign
            | Self::MulAssign
            | Self::DivAssign
            | Self::RemAssign
            | Self::AndAssign
            | Self::OrAssign
            | Self::BitAndAssign
            | Self::BitOrAssign
            | Self::BitXorAssign => false,
        }
    }

    fn is_right_associative(&self) -> bool {
        !self.is_left_associative()
    }
}

fn token(kind: TokenKind) -> impl Parser<Token, Token, Error = Error> {
    just(Token::of(kind))
}

fn non_padded(kind: TokenKind) -> impl Parser<Token, Token, Error = Error> {
    just(Token {
        kind,
        has_space_before: Some(false),
    })
}

fn comma_separated<P: 'static, T: 'static>(parser: P) -> impl Parser<Token, Vec<T>, Error = Error>
where
    P: Parser<Token, T, Error = Error>,
{
    // Parse a set of items separated by commas, allowing for a trailing comma.
    recursive(|tail| {
        parser
            .chain(
                token(TokenKind::Comma)
                    .padding_for(tail.or_not().flatten())
                    .or_not()
                    .flatten(),
            )
            .or_not()
            .flatten()
    })
}

fn binary_op() -> impl Parser<Token, BinaryOp, Error = Error> {
    (token(TokenKind::Plus)
        .then(
            non_padded(TokenKind::Equal)
                .to(BinaryOp::AddAssign)
                .or_not(),
        )
        .map(|(_, op)| op.unwrap_or(BinaryOp::Add)))
    .or(token(TokenKind::Minus)
        .to(BinaryOp::Sub)
        .then(
            non_padded(TokenKind::Equal)
                .to(BinaryOp::SubAssign)
                .or_not(),
        )
        .map(|(_, op)| op.unwrap_or(BinaryOp::Sub)))
    .or(token(TokenKind::Asterisk)
        .then(
            non_padded(TokenKind::Equal)
                .to(BinaryOp::MulAssign)
                .or_not(),
        )
        .map(|(_, op)| op.unwrap_or(BinaryOp::Mul)))
    .or(token(TokenKind::Slash)
        .then(
            non_padded(TokenKind::Equal)
                .to(BinaryOp::DivAssign)
                .or_not(),
        )
        .map(|(_, op)| op.unwrap_or(BinaryOp::Div)))
    .or(token(TokenKind::Percent)
        .then(
            non_padded(TokenKind::Equal)
                .to(BinaryOp::RemAssign)
                .or_not(),
        )
        .map(|(_, op)| op.unwrap_or(BinaryOp::Rem)))
    .or(token(TokenKind::And)
        .then(non_padded(TokenKind::And).or_not())
        .then(non_padded(TokenKind::Equal).or_not())
        .map(|((_, and), assign)| match (and, assign) {
            (None, None) => BinaryOp::BitAnd,
            (Some(_), None) => BinaryOp::And,
            (None, Some(_)) => BinaryOp::BitAndAssign,
            (Some(_), Some(_)) => BinaryOp::AndAssign,
        }))
    .or(token(TokenKind::Pipe)
        .then(non_padded(TokenKind::Pipe).or_not())
        .then(non_padded(TokenKind::Equal).or_not())
        .map(|((_, or), assign)| match (or, assign) {
            (None, None) => BinaryOp::BitOr,
            (Some(_), None) => BinaryOp::Or,
            (None, Some(_)) => BinaryOp::BitOrAssign,
            (Some(_), Some(_)) => BinaryOp::OrAssign,
        }))
    .or(token(TokenKind::Caret)
        .then(
            non_padded(TokenKind::Equal)
                .to(BinaryOp::BitXorAssign)
                .or_not(),
        )
        .map(|(_, op)| op.unwrap_or(BinaryOp::BitXor)))
    .or(token(TokenKind::Equal)
        .then(non_padded(TokenKind::Equal).to(BinaryOp::Eq).or_not())
        .map(|(_, op)| op.unwrap_or(BinaryOp::Assign)))
    .or(token(TokenKind::Exclamation)
        .then(non_padded(TokenKind::Equal))
        .to(BinaryOp::Ne))
    .or(token(TokenKind::Less)
        .then(non_padded(TokenKind::Equal).to(BinaryOp::Le).or_not())
        .map(|(_, op)| op.unwrap_or(BinaryOp::Lt)))
    .or(token(TokenKind::Greater)
        .then(non_padded(TokenKind::Greater).to(BinaryOp::Ge).or_not())
        .map(|(_, op)| op.unwrap_or(BinaryOp::Gt)))
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Pos,
    Neg,
    Not,
    Call(Vec<Expr>),
    Index(Expr),
    Member(String),
}

fn prefix_op() -> impl Parser<Token, UnaryOp, Error = Error> {
    (token(TokenKind::Plus).to(UnaryOp::Pos))
        .or(token(TokenKind::Minus).to(UnaryOp::Neg))
        .or(token(TokenKind::Exclamation).to(UnaryOp::Not))
}

fn postfix_op<E>(expr: E) -> impl Parser<Token, UnaryOp, Error = Error>
where
    E: Parser<Token, Expr, Error = Error> + Clone + 'static,
{
    (comma_separated(expr.clone())
        .delimited_by(
            Token::of(TokenKind::LeftParen),
            Token::of(TokenKind::RightParen),
        )
        .map(|option| option.unwrap_or_else(|| vec![Expr::Invalid]))
        .map(UnaryOp::Call))
    .or((expr.clone())
        .delimited_by(
            Token::of(TokenKind::LeftBracket),
            Token::of(TokenKind::RightBracket),
        )
        .map(|option| option.unwrap_or(Expr::Invalid))
        .map(UnaryOp::Index))
    .or(
        token(TokenKind::Dot).padding_for(filter_map(|span, token: Token| match token.kind {
            TokenKind::Identifier(name) => Ok(UnaryOp::Member(name)),
            _ => Err(Error::expected_token_found(
                Some(span),
                vec![Token::of(TokenKind::Identifier("..".to_string()))],
                Some(token),
            )),
        })),
    )
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Invalid,
    Literal(Value),
    Variable(String),
    BinaryOp(Box<(BinaryOp, Expr, Expr)>),
    UnaryOp(Box<(UnaryOp, Expr)>),
}

fn term<E>(expr: E) -> impl Parser<Token, Expr, Error = Error>
where
    E: Parser<Token, Expr, Error = Error> + Clone + 'static,
{
    // Subset of the expression parser that parses anything except binary operators.
    // This is used to prevent the sub-expressions of a binary operator from parsing
    // operators that we want the parent to parse.
    (prefix_op().repeated())
        .then(
            filter_map(|span, token: Token| match token.kind {
                TokenKind::Identifier(name) => Ok(Expr::Variable(name)),
                TokenKind::True => Ok(Expr::Literal(Value::Bool(true))),
                TokenKind::False => Ok(Expr::Literal(Value::Bool(false))),
                TokenKind::Int(value) => Ok(Expr::Literal(Value::Int(value))),
                TokenKind::Float(value) => Ok(Expr::Literal(Value::Float(value))),
                TokenKind::String(value) => Ok(Expr::Literal(Value::String(value))),
                _ => Err(Error::expected_token_found(Some(span), vec![], Some(token))),
            })
            // Allow parsing a full expression if it is enclosed in parentheses.
            .or((expr.clone())
                .delimited_by(
                    Token::of(TokenKind::LeftParen),
                    Token::of(TokenKind::RightParen),
                )
                .map(|option| option.unwrap_or(Expr::Invalid))),
        )
        .then(postfix_op(expr).repeated())
        .map(|((prefix, base), postfix)| {
            let mut ops = postfix;
            ops.extend(prefix.into_iter().rev());
            let mut expr = base;
            for op in ops {
                expr = Expr::UnaryOp(Box::new((op, expr)));
            }
            expr
        })
}

struct ShuntingYard {
    output_stack: Vec<Expr>,
    op_stack: Vec<BinaryOp>,
}

impl ShuntingYard {
    fn new() -> Self {
        Self {
            output_stack: Vec::new(),
            op_stack: Vec::new(),
        }
    }

    fn push_expr(&mut self, expr: Expr) {
        self.output_stack.push(expr)
    }

    fn push_op(&mut self, op: BinaryOp) {
        while let Some(head) = self.op_stack.last() {
            if head.precedence() <= op.precedence()
                && (op.is_right_associative() || head.precedence() != op.precedence())
            {
                break;
            }
            self.reduce();
        }
        self.op_stack.push(op);
    }

    fn reduce(&mut self) {
        let op = self.op_stack.pop().expect("stack underflow");
        let right = self.output_stack.pop().expect("stack underflow");
        let left = self.output_stack.pop().expect("stack underflow");
        self.output_stack
            .push(Expr::BinaryOp(Box::new((op, left, right))));
    }

    fn finish(mut self) -> Expr {
        while !self.op_stack.is_empty() {
            self.reduce();
        }
        assert!(self.output_stack.len() == 1, "output not fully reduced");
        self.output_stack.pop().unwrap()
    }
}

pub fn expr() -> impl Parser<Token, Expr, Error = Error> {
    recursive(|expr| {
        (term(expr.clone()))
            // Handle trailing binary operators
            .then(binary_op().then(term(expr)).repeated())
            .map(|(base, ops)| {
                if ops.is_empty() {
                    base
                } else {
                    let mut shunting_yard = ShuntingYard::new();
                    shunting_yard.push_expr(base);
                    for (op, term) in ops {
                        shunting_yard.push_op(op);
                        shunting_yard.push_expr(term);
                    }
                    shunting_yard.finish()
                }
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::token::get_tokens;

    fn assert_expr(input: &str, expected: Expr) {
        let tokens = get_tokens(input).unwrap();
        let parsed = expr().padded_by(end()).parse(tokens).unwrap();
        assert_eq!(expected, parsed);
    }

    #[test]
    fn two_plus_two() {
        assert_expr(
            "2 + 2",
            Expr::BinaryOp(Box::new((
                BinaryOp::Add,
                Expr::Literal(Value::Int(2)),
                Expr::Literal(Value::Int(2)),
            ))),
        );
    }

    #[test]
    fn multiple_operators() {
        assert_expr(
            "1 - 2 + -3 - -!4 * 5 / 6 % 7",
            Expr::BinaryOp(Box::new((
                BinaryOp::Sub,
                Expr::BinaryOp(Box::new((
                    BinaryOp::Add,
                    Expr::BinaryOp(Box::new((
                        BinaryOp::Sub,
                        Expr::Literal(Value::Int(1)),
                        Expr::Literal(Value::Int(2)),
                    ))),
                    Expr::UnaryOp(Box::new((UnaryOp::Neg, Expr::Literal(Value::Int(3))))),
                ))),
                Expr::BinaryOp(Box::new((
                    BinaryOp::Rem,
                    Expr::BinaryOp(Box::new((
                        BinaryOp::Div,
                        Expr::BinaryOp(Box::new((
                            BinaryOp::Mul,
                            Expr::UnaryOp(Box::new((
                                UnaryOp::Neg,
                                Expr::UnaryOp(Box::new((
                                    UnaryOp::Not,
                                    Expr::Literal(Value::Int(4)),
                                ))),
                            ))),
                            Expr::Literal(Value::Int(5)),
                        ))),
                        Expr::Literal(Value::Int(6)),
                    ))),
                    Expr::Literal(Value::Int(7)),
                ))),
            ))),
        );
    }

    #[test]
    fn parentheses() {
        assert_expr(
            "(1 + 2) + (3 * -(4 * 5 - 6))",
            Expr::BinaryOp(Box::new((
                BinaryOp::Add,
                Expr::BinaryOp(Box::new((
                    BinaryOp::Add,
                    Expr::Literal(Value::Int(1)),
                    Expr::Literal(Value::Int(2)),
                ))),
                Expr::BinaryOp(Box::new((
                    BinaryOp::Mul,
                    Expr::Literal(Value::Int(3)),
                    Expr::UnaryOp(Box::new((
                        UnaryOp::Neg,
                        Expr::BinaryOp(Box::new((
                            BinaryOp::Sub,
                            Expr::BinaryOp(Box::new((
                                BinaryOp::Mul,
                                Expr::Literal(Value::Int(4)),
                                Expr::Literal(Value::Int(5)),
                            ))),
                            Expr::Literal(Value::Int(6)),
                        ))),
                    ))),
                ))),
            ))),
        )
    }

    #[test]
    fn function_call() {
        assert_expr(
            "foo(bar, baz)",
            Expr::UnaryOp(Box::new((
                UnaryOp::Call(vec![
                    Expr::Variable("bar".to_string()),
                    Expr::Variable("baz".to_string()),
                ]),
                Expr::Variable("foo".to_string()),
            ))),
        )
    }

    #[test]
    fn function_call_with_prefixes() {
        assert_expr(
            "!is_even(5)",
            Expr::UnaryOp(Box::new((
                UnaryOp::Not,
                Expr::UnaryOp(Box::new((
                    UnaryOp::Call(vec![Expr::Literal(Value::Int(5))]),
                    Expr::Variable("is_even".to_string()),
                ))),
            ))),
        )
    }

    #[test]
    fn function_call_trailing_comma() {
        assert_expr(
            "foo(bar,)",
            Expr::UnaryOp(Box::new((
                UnaryOp::Call(vec![Expr::Variable("bar".to_string())]),
                Expr::Variable("foo".to_string()),
            ))),
        )
    }

    #[test]
    fn function_call_no_args() {
        assert_expr(
            "foo()",
            Expr::UnaryOp(Box::new((
                UnaryOp::Call(vec![]),
                Expr::Variable("foo".to_string()),
            ))),
        )
    }

    #[test]
    fn index() {
        assert_expr(
            "arr[2]",
            Expr::UnaryOp(Box::new((
                UnaryOp::Index(Expr::Literal(Value::Int(2))),
                Expr::Variable("arr".to_string()),
            ))),
        )
    }

    #[test]
    fn member_access() {
        assert_expr(
            "object.method()",
            Expr::UnaryOp(Box::new((
                UnaryOp::Call(vec![]),
                Expr::UnaryOp(Box::new((
                    UnaryOp::Member("method".to_string()),
                    Expr::Variable("object".to_string()),
                ))),
            ))),
        )
    }

    #[test]
    fn assign() {
        assert_expr(
            "j = i += 1",
            Expr::BinaryOp(Box::new((
                BinaryOp::Assign,
                Expr::Variable("j".to_string()),
                Expr::BinaryOp(Box::new((
                    BinaryOp::AddAssign,
                    Expr::Variable("i".to_string()),
                    Expr::Literal(Value::Int(1)),
                ))),
            ))),
        )
    }
}
