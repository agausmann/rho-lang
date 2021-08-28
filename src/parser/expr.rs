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
    (token(TokenKind::Plus).to(BinaryOp::Add))
        .or(token(TokenKind::Minus).to(BinaryOp::Sub))
        .or(token(TokenKind::Asterisk).to(BinaryOp::Mul))
        .or(token(TokenKind::Slash).to(BinaryOp::Div))
        .or(token(TokenKind::Percent).to(BinaryOp::Mod))
        .or(token(TokenKind::And)
            .then(non_padded(TokenKind::And).to(BinaryOp::And).or_not())
            .map(|(_, op)| op.unwrap_or(BinaryOp::BitAnd)))
        .or(token(TokenKind::Pipe)
            .then(non_padded(TokenKind::Pipe).to(BinaryOp::Or).or_not())
            .map(|(_, op)| op.unwrap_or(BinaryOp::BitOr)))
        .or(token(TokenKind::Caret).to(BinaryOp::BitXor))
        .or(token(TokenKind::Equal)
            .then(non_padded(TokenKind::Equal))
            .to(BinaryOp::Eq))
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
pub enum PrefixOp {
    Pos,
    Neg,
    Not,
}

fn prefix_op() -> impl Parser<Token, PrefixOp, Error = Error> {
    (token(TokenKind::Plus).to(PrefixOp::Pos))
        .or(token(TokenKind::Minus).to(PrefixOp::Neg))
        .or(token(TokenKind::Exclamation).to(PrefixOp::Not))
}

#[derive(Debug, Clone, PartialEq)]
pub enum PostfixOp {
    Call(Vec<Expr>),
    Index(Expr),
}

fn postfix_op<E>(expr: E) -> impl Parser<Token, PostfixOp, Error = Error>
where
    E: Parser<Token, Expr, Error = Error> + Clone + 'static,
{
    (comma_separated(expr.clone())
        .delimited_by(
            Token::of(TokenKind::LeftParen),
            Token::of(TokenKind::RightParen),
        )
        .map(|option| option.unwrap_or_else(|| vec![Expr::Invalid]))
        .map(PostfixOp::Call))
    .or((expr.clone())
        .delimited_by(
            Token::of(TokenKind::LeftBracket),
            Token::of(TokenKind::RightBracket),
        )
        .map(|option| option.unwrap_or(Expr::Invalid))
        .map(PostfixOp::Index))
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
            "foo(bar, baz)",
            Expr::Term(
                vec![],
                Box::new(Expr::Variable("foo".to_string())),
                vec![PostfixOp::Call(vec![
                    Expr::Variable("bar".to_string()),
                    Expr::Variable("baz".to_string()),
                ])],
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
                vec![PostfixOp::Call(vec![Expr::Literal(Value::Int(5))])],
            ),
        )
    }

    #[test]
    fn function_call_trailing_comma() {
        assert_expr(
            "foo(bar,)",
            Expr::Term(
                vec![],
                Box::new(Expr::Variable("foo".to_string())),
                vec![PostfixOp::Call(vec![Expr::Variable("bar".to_string())])],
            ),
        )
    }

    #[test]
    fn function_call_no_args() {
        assert_expr(
            "foo()",
            Expr::Term(
                vec![],
                Box::new(Expr::Variable("foo".to_string())),
                vec![PostfixOp::Call(vec![])],
            ),
        )
    }

    #[test]
    fn index() {
        assert_expr(
            "arr[2]",
            Expr::Term(
                vec![],
                Box::new(Expr::Variable("arr".to_string())),
                vec![PostfixOp::Index(Expr::Literal(Value::Int(2)))],
            ),
        )
    }
}
