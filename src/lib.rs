use chumsky::prelude::*;

const WHITESPACE: &str = " \t\n\r";
const ALPHA_UND: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_";
const ALPHA_NUM_UND: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_";
const NONZERO_DIGIT: &str = "123456789";
const DIGIT: &str = "0123456789";
const HEXDIGIT: &str = "0123456789ABCDEFabcdef";
const BINDIGIT: &str = "01";

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    String(String),
    Int(i64),
    Float(f64),
}

type Error<T> = Simple<T>;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Whitespace,
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    LeftBrace,
    RightBrace,
    Exclamation,
    Plus,
    Minus,
    Asterisk,
    Slash,
    Percent,
    And,
    Pipe,
    Caret,
    Dot,
    Comma,
    Colon,
    Equal,
    Less,
    Greater,
    Identifier(String),
    Int(i64),
    Float(f64),
    String(String),
}

fn nonzero_digit() -> impl Parser<char, char, Error = Error<char>> {
    one_of(NONZERO_DIGIT.chars())
}

fn digit() -> impl Parser<char, char, Error = Error<char>> {
    one_of(DIGIT.chars())
}

fn hexdigit() -> impl Parser<char, char, Error = Error<char>> {
    one_of(HEXDIGIT.chars())
}

fn bindigit() -> impl Parser<char, char, Error = Error<char>> {
    one_of(BINDIGIT.chars())
}

fn float_exponent() -> impl Parser<char, Vec<char>, Error = Error<char>> {
    one_of("eE".chars())
        .chain(one_of("+-".chars()).or_not())
        .chain(digit().repeated_at_least(1))
}

fn float_after_decimal() -> impl Parser<char, Vec<char>, Error = Error<char>> {
    digit().repeated_at_least(1)
}

fn float_int_suffix() -> impl Parser<char, Vec<char>, Error = Error<char>> {
    (just('.')
        .chain(float_after_decimal().or_not().flatten())
        .or_not()
        .flatten())
    .chain(float_exponent().or_not().flatten())
}

fn parse_number(string: String) -> Token {
    if string.contains(&['.', 'e', 'E'][..]) {
        Token::Float(string.parse().unwrap())
    } else {
        Token::Int(string.parse().unwrap())
    }
}

//TODO: Floats starting with a decimal point (.)
fn leading_digit_number() -> impl Parser<char, Token, Error = Error<char>> {
    (nonzero_digit()
        .chain(digit().repeated())
        .chain::<char, _, _>(float_int_suffix())
        .collect::<String>()
        .map(parse_number))
    .or(just('0').padding_for(
        (just('x').padding_for(
            hexdigit()
                .repeated_at_least(1)
                .collect::<String>()
                .map(|s| Token::Int(i64::from_str_radix(&s, 16).unwrap())),
        ))
        .or(just('b').padding_for(
            bindigit()
                .repeated_at_least(1)
                .collect::<String>()
                .map(|s| Token::Int(i64::from_str_radix(&s, 2).unwrap())),
        ))
        .or(digit()
            .repeated_at_least(1)
            .chain::<char, _, _>(float_int_suffix())
            .collect::<String>()
            .map(parse_number)),
    ))
}

fn string_contents() -> impl Parser<char, Token, Error = Error<char>> {
    (none_of("\"\\".chars()).map(Some))
        .or(just('\\').padding_for(
            // Escape codes
            (just('n').to('\n'))
                .or(just('r').to('\r'))
                .or(just('t').to('\t'))
                .or(just('0').to('\0'))
                .or(just('\\').to('\\'))
                .or(just('"').to('"'))
                .or(just('x')
                    .padding_for(hexdigit().then(hexdigit()))
                    .map(|(high, low)| {
                        ((high.to_digit(16).unwrap() << 4) | low.to_digit(16).unwrap()) as u8
                            as char
                    }))
                .map(Some)
                // Ignore all whitespace (including newlines) after a backslash
                .or(one_of(WHITESPACE.chars()).repeated_at_least(1).to(None)),
        ))
        .repeated()
        .flatten()
        .collect()
        .map(Token::String)
}

fn tokenizer() -> impl Parser<char, Token, Error = Error<char>> {
    (one_of(WHITESPACE.chars())
        .repeated_at_least(1)
        .to(Token::Whitespace))
    .or(just('(').to(Token::LeftParen))
    .or(just(')').to(Token::RightParen))
    .or(just('[').to(Token::LeftBracket))
    .or(just(']').to(Token::RightBracket))
    .or(just('{').to(Token::LeftBrace))
    .or(just('}').to(Token::RightBrace))
    .or(just('!').to(Token::Exclamation))
    .or(just('+').to(Token::Plus))
    .or(just('-').to(Token::Minus))
    .or(just('*').to(Token::Asterisk))
    .or(just('/')
        .then(
            // Single-line comment
            (just('/')
                // Take until end of line
                .then(none_of("\n\r".chars()).repeated())
                .to(Token::Whitespace))
            // Multiline comment
            .or(just('*')
                // Zero or more characters that does not contain an asterisk.
                .then(none_of("*".chars()).repeated())
                // Trailing asterisk.
                .then(just('*'))
                // Zero or more sequences of...
                .then(
                    // Zero or more characters that does not start with a slash
                    // and that does not contain an asterisk.
                    (none_of("/*".chars())
                        .then(none_of("*".chars()).repeated())
                        .or_not())
                    // Trailing asterisk.
                    .then(just('*'))
                    .repeated(),
                )
                // Trailing slash.
                .then(just('/'))
                .to(Token::Whitespace))
            .or_not(),
        )
        .map(|(_, comment)| comment.unwrap_or(Token::Slash)))
    .or(just('%').to(Token::Percent))
    .or(just('&').to(Token::And))
    .or(just('|').to(Token::Pipe))
    .or(just('^').to(Token::Caret))
    .or(just('.').to(Token::Dot))
    .or(just(',').to(Token::Comma))
    .or(just(':').to(Token::Colon))
    .or(just('=').to(Token::Equal))
    .or(just('<').to(Token::Less))
    .or(just('>').to(Token::Greater))
    .or(one_of(ALPHA_UND.chars())
        .chain(one_of(ALPHA_NUM_UND.chars()).repeated())
        .collect()
        .map(|s| match s {
            // TODO Check for keywords
            _ => Token::Identifier(s),
        }))
    .or(leading_digit_number())
    .or(just('"')
        .padding_for(string_contents())
        .padded_by(just('"')))
}

pub fn get_tokens(s: &str) -> Result<Vec<Token>, Vec<Error<char>>> {
    tokenizer().repeated().padded_by(end()).parse(s)
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

fn binary_op() -> impl Parser<Token, BinaryOp, Error = Error<Token>> {
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

fn unary_op() -> impl Parser<Token, UnaryOp, Error = Error<Token>> {
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

pub fn expr() -> impl Parser<Token, Expr, Error = Error<Token>> {
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

    fn assert_tokens(input: &str) {
        dbg!(get_tokens(input).unwrap());
    }

    #[test]
    fn comment() {
        assert_tokens("// This is a line_comment");
    }

    #[test]
    fn block_comment() {
        assert_tokens("/* This is a block comment */");
    }

    #[test]
    fn block_comment_with_asterisk() {
        assert_tokens("/** This *is* a block comment with embedded asterisks. **/");
    }

    #[test]
    fn string() {
        assert_tokens(r#" "Hello World" "#);
    }

    #[test]
    fn string_with_escapes() {
        assert_tokens(r#" "Hello\r\n\    \t \"World!\"\x0A" "#);
    }

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
