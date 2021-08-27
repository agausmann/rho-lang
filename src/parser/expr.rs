use chumsky::prelude::*;
use crate::parser::token::Token;

pub type Error = Simple<Token>;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    String(String),
    Int(i64),
    Float(f64),
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
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
}

fn binary_op() -> impl Parser<Token, BinaryOp, Error = Error> {
    (just(Token::Plus).to(BinaryOp::Add))
        .or(just(Token::Minus).to(BinaryOp::Sub))
        .or(just(Token::Asterisk).to(BinaryOp::Mul))
        .or(just(Token::Slash).to(BinaryOp::Div))
        .or(just(Token::Percent).to(BinaryOp::Mod))
        .or(just(Token::And)
            .then(just(Token::And).to(BinaryOp::And).or_not())
            .map(|(_, op)| op.unwrap_or(BinaryOp::BitAnd)))
        .or(just(Token::Pipe)
            .then(just(Token::Pipe).to(BinaryOp::Or).or_not())
            .map(|(_, op)| op.unwrap_or(BinaryOp::BitOr)))
        .or(just(Token::Caret).to(BinaryOp::BitXor))
        .or(just(Token::Equal).then(just(Token::Equal)).to(BinaryOp::Eq))
        .or(just(Token::Exclamation)
            .then(just(Token::Equal))
            .to(BinaryOp::Ne))
        .or(just(Token::Less)
            .then(just(Token::Equal).to(BinaryOp::Le).or_not())
            .map(|(_, op)| op.unwrap_or(BinaryOp::Lt)))
        .or(just(Token::Greater)
            .then(just(Token::Greater).to(BinaryOp::Ge).or_not())
            .map(|(_, op)| op.unwrap_or(BinaryOp::Gt)))
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Pos,
    Neg,
    Not,
}

fn unary_op() -> impl Parser<Token, UnaryOp, Error = Error> {
    (just(Token::Plus).to(UnaryOp::Pos))
        .or(just(Token::Minus).to(UnaryOp::Neg))
        .or(just(Token::Exclamation).to(UnaryOp::Not))
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Invalid,
    Literal(Value),
    Variable(String),
    BinaryOps(Box<Expr>, Vec<(BinaryOp, Expr)>),
    UnaryOps(Box<Expr>, Vec<UnaryOp>),
}

pub fn expr() -> impl Parser<Token, Expr, Error = Error> {
    recursive(|expr| {
        // Sub-parser that parses anything except binary operators.
        // This is used to prevent the sub-expressions of a binary operator from parsing
        // operators that we want the parent to parse.
        let term = recursive(|term| {
            (just::<_, Simple<Token>>(Token::Whitespace).repeated()).padding_for(
                filter_map(|span, token| match token {
                    Token::Identifier(name) => Ok(Expr::Variable(name)),
                    Token::String(value) => Ok(Expr::Literal(Value::String(value))),
                    Token::Int(value) => Ok(Expr::Literal(Value::Int(value))),
                    Token::Float(value) => Ok(Expr::Literal(Value::Float(value))),
                    _ => Err(Error::expected_token_found(Some(span), vec![], Some(token))),
                })
                // Allow parsing a full expression if it is enclosed in parentheses.
                .or((expr.clone())
                    .delimited_by(Token::LeftParen, Token::RightParen)
                    .map(|option| option.unwrap_or(Expr::Invalid)))
                .or(unary_op()
                    .repeated_at_least(1)
                    .then(term)
                    .map(|(ops, base)| Expr::UnaryOps(Box::new(base), ops)))
                // Handle trailing whitespace
                .padded_by(just(Token::Whitespace).repeated()),
            )
        });

        (term.clone())
            // Handle trailing binary operators
            .then(
                binary_op()
                    .then(term.clone())
                    .repeated_at_least(1)
                    .or_not(),
            )
            .map(|(base, maybe_ops)| match maybe_ops {
                Some(ops) => Expr::BinaryOps(Box::new(base), ops),
                None => base,
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
            Expr::BinaryOps(
                Box::new(Expr::Literal(Value::Int(2))),
                vec![(BinaryOp::Add, Expr::Literal(Value::Int(2)))],
            ),
        );
    }

    #[test]
    fn multiple_operators() {
        assert_expr(
            "1 - 2 + -3 - -!4 * 5 / 6 % 7",
            Expr::BinaryOps(
                Box::new(Expr::Literal(Value::Int(1))),
                vec![
                    (BinaryOp::Sub, Expr::Literal(Value::Int(2))),
                    (BinaryOp::Add, Expr::UnaryOps(Box::new(Expr::Literal(Value::Int(3))), vec![UnaryOp::Neg])),
                    (BinaryOp::Sub, Expr::UnaryOps(Box::new(Expr::Literal(Value::Int(4))), vec![UnaryOp::Neg, UnaryOp::Not])),
                    (BinaryOp::Mul, Expr::Literal(Value::Int(5))),
                    (BinaryOp::Div, Expr::Literal(Value::Int(6))),
                    (BinaryOp::Mod, Expr::Literal(Value::Int(7))),
                ],
            ),
        );
    }
}
