use crate::parser::token::Token;
use chumsky::prelude::*;

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
pub enum PrefixOp {
    Pos,
    Neg,
    Not,
}

fn prefix_op() -> impl Parser<Token, PrefixOp, Error = Error> {
    (just(Token::Plus).to(PrefixOp::Pos))
        .or(just(Token::Minus).to(PrefixOp::Neg))
        .or(just(Token::Exclamation).to(PrefixOp::Not))
}

#[derive(Debug, Clone, PartialEq)]
pub enum PostfixOp {
    Call(Vec<Expr>),
}

fn postfix_op(
    expr: impl Parser<Token, Expr, Error = Error>,
) -> impl Parser<Token, PostfixOp, Error = Error> {
    // Not allowed to recursively call `expr`.
    expr.repeated()
        .delimited_by(Token::LeftParen, Token::RightParen)
        .map(|option| option.unwrap_or_else(|| vec![Expr::Invalid]))
        .map(PostfixOp::Call)
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Invalid,
    Literal(Value),
    Variable(String),
    BinaryOps(Box<Expr>, Vec<(BinaryOp, Expr)>),
    Term(Vec<PrefixOp>, Box<Expr>, Vec<PostfixOp>),
}

fn term<E>(expr: E) -> impl Parser<Token, Expr, Error = Error>
where
    E: Parser<Token, Expr, Error = Error> + Clone,
{
    // Subset of the expression parser that parses anything except binary operators.
    // This is used to prevent the sub-expressions of a binary operator from parsing
    // operators that we want the parent to parse.
    (just::<_, Simple<Token>>(Token::Whitespace).repeated())
        .padding_for(
            (prefix_op().repeated())
                .then(
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
                        .map(|option| option.unwrap_or(Expr::Invalid))),
                )
                .then(postfix_op(expr).repeated()),
        )
        // Handle trailing whitespace
        .padded_by(just(Token::Whitespace).repeated())
        .map(|((prefix, base), postfix)| {
            if prefix.is_empty() && postfix.is_empty() {
                base
            } else {
                Expr::Term(prefix, Box::new(base), postfix)
            }
        })
}

pub fn expr() -> impl Parser<Token, Expr, Error = Error> {
    recursive(|expr| {
        (term(expr.clone()))
            // Handle trailing binary operators
            .then(binary_op().then(term(expr)).repeated_at_least(1).or_not())
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
                    (
                        BinaryOp::Add,
                        Expr::Term(
                            vec![PrefixOp::Neg],
                            Box::new(Expr::Literal(Value::Int(3))),
                            vec![],
                        ),
                    ),
                    (
                        BinaryOp::Sub,
                        Expr::Term(
                            vec![PrefixOp::Neg, PrefixOp::Not],
                            Box::new(Expr::Literal(Value::Int(4))),
                            vec![],
                        ),
                    ),
                    (BinaryOp::Mul, Expr::Literal(Value::Int(5))),
                    (BinaryOp::Div, Expr::Literal(Value::Int(6))),
                    (BinaryOp::Mod, Expr::Literal(Value::Int(7))),
                ],
            ),
        );
    }

    #[test]
    fn parentheses() {
        assert_expr(
            "(1 + 2) + (3 * -(4 * 5 - 6))",
            Expr::BinaryOps(
                Box::new(Expr::BinaryOps(
                    Box::new(Expr::Literal(Value::Int(1))),
                    vec![(BinaryOp::Add, Expr::Literal(Value::Int(2)))],
                )),
                vec![(
                    BinaryOp::Add,
                    Expr::BinaryOps(
                        Box::new(Expr::Literal(Value::Int(3))),
                        vec![(
                            BinaryOp::Mul,
                            Expr::Term(
                                vec![PrefixOp::Neg],
                                Box::new(Expr::BinaryOps(
                                    Box::new(Expr::Literal(Value::Int(4))),
                                    vec![
                                        (BinaryOp::Mul, Expr::Literal(Value::Int(5))),
                                        (BinaryOp::Sub, Expr::Literal(Value::Int(6))),
                                    ],
                                )),
                                vec![],
                            ),
                        )],
                    ),
                )],
            ),
        )
    }

    #[test]
    fn function_call() {
        assert_expr(
            "foo(bar)",
            Expr::Term(
                vec![],
                Box::new(Expr::Variable("foo".to_string())),
                vec![PostfixOp::Call(vec![Expr::Variable("bar".to_string())])],
            ),
        )
    }

    #[test]
    fn function_call_with_prefixes() {
        assert_expr(
            "!is_even(5)",
            Expr::Term(
                vec![PrefixOp::Not],
                Box::new(Expr::Variable("is_even".to_string())),
                vec![PostfixOp::Call(vec![Expr::Literal(Value::Int(5))])]
            )
        )
    }
}
