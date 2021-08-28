use chumsky::prelude::*;

const WHITESPACE: &str = " \t\n\r";
const ALPHA_UND: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_";
const ALPHA_NUM_UND: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_";
const NONZERO_DIGIT: &str = "123456789";
const DIGIT: &str = "0123456789";
const HEXDIGIT: &str = "0123456789ABCDEFabcdef";
const BINDIGIT: &str = "01";

pub type Error = Simple<char>;

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub has_space_before: Option<bool>,
}

impl Token {
    pub fn of(kind: TokenKind) -> Self {
        Self {
            kind,
            has_space_before: None,
        }
    }
}

// Custom PartialEq to only compare `has_space_before` if both are `Some`.
// Enables creating `Token` values to match against only the `TokenKind` where whitespace doesn't
// matter, by setting `has_space_before` to `None`.
impl PartialEq for Token {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
            && (self.has_space_before.is_none()
                || other.has_space_before.is_none()
                || self.has_space_before == other.has_space_before)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
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
    True,
    False,
}

fn nonzero_digit() -> impl Parser<char, char, Error = Error> {
    one_of(NONZERO_DIGIT.chars())
}

fn digit() -> impl Parser<char, char, Error = Error> {
    one_of(DIGIT.chars())
}

fn hexdigit() -> impl Parser<char, char, Error = Error> {
    one_of(HEXDIGIT.chars())
}

fn bindigit() -> impl Parser<char, char, Error = Error> {
    one_of(BINDIGIT.chars())
}

fn float_exponent() -> impl Parser<char, Vec<char>, Error = Error> {
    one_of("eE".chars())
        .chain(one_of("+-".chars()).or_not())
        .chain(digit().repeated_at_least(1))
}

fn float_after_decimal() -> impl Parser<char, Vec<char>, Error = Error> {
    digit().repeated_at_least(1)
}

fn float_int_suffix() -> impl Parser<char, Vec<char>, Error = Error> {
    (just('.')
        .chain(float_after_decimal().or_not().flatten())
        .or_not()
        .flatten())
    .chain(float_exponent().or_not().flatten())
}

fn parse_number(string: String) -> TokenKind {
    if string.contains(&['.', 'e', 'E'][..]) {
        TokenKind::Float(string.parse().unwrap())
    } else {
        TokenKind::Int(string.parse().unwrap())
    }
}

//TODO: Floats starting with a decimal point (.)
fn leading_digit_number() -> impl Parser<char, TokenKind, Error = Error> {
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
                .map(|s| TokenKind::Int(i64::from_str_radix(&s, 16).unwrap())),
        ))
        .or(just('b').padding_for(
            bindigit()
                .repeated_at_least(1)
                .collect::<String>()
                .map(|s| TokenKind::Int(i64::from_str_radix(&s, 2).unwrap())),
        ))
        .or(digit()
            .repeated_at_least(1)
            .chain::<char, _, _>(float_int_suffix())
            .collect::<String>()
            .map(parse_number)),
    ))
}

fn string_contents() -> impl Parser<char, TokenKind, Error = Error> {
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
        .map(TokenKind::String)
}

fn token_kind() -> impl Parser<char, TokenKind, Error = Error> {
    (one_of(WHITESPACE.chars())
        .repeated_at_least(1)
        .to(TokenKind::Whitespace))
    .or(just('(').to(TokenKind::LeftParen))
    .or(just(')').to(TokenKind::RightParen))
    .or(just('[').to(TokenKind::LeftBracket))
    .or(just(']').to(TokenKind::RightBracket))
    .or(just('{').to(TokenKind::LeftBrace))
    .or(just('}').to(TokenKind::RightBrace))
    .or(just('!').to(TokenKind::Exclamation))
    .or(just('+').to(TokenKind::Plus))
    .or(just('-').to(TokenKind::Minus))
    .or(just('*').to(TokenKind::Asterisk))
    .or(just('/')
        .then(
            // Single-line comment
            (just('/')
                // Take until end of line
                .then(none_of("\n\r".chars()).repeated())
                .to(TokenKind::Whitespace))
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
                .to(TokenKind::Whitespace))
            .or_not(),
        )
        .map(|(_, comment)| comment.unwrap_or(TokenKind::Slash)))
    .or(just('%').to(TokenKind::Percent))
    .or(just('&').to(TokenKind::And))
    .or(just('|').to(TokenKind::Pipe))
    .or(just('^').to(TokenKind::Caret))
    .or(just('.').to(TokenKind::Dot))
    .or(just(',').to(TokenKind::Comma))
    .or(just(':').to(TokenKind::Colon))
    .or(just('=').to(TokenKind::Equal))
    .or(just('<').to(TokenKind::Less))
    .or(just('>').to(TokenKind::Greater))
    .or(one_of(ALPHA_UND.chars())
        .chain(one_of(ALPHA_NUM_UND.chars()).repeated())
        .collect::<String>()
        .map(|s| match &*s {
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            _ => TokenKind::Identifier(s),
        }))
    .or(leading_digit_number())
    .or(just('"')
        .padding_for(string_contents())
        .padded_by(just('"')))
}

fn tokens() -> impl Parser<TokenKind, Vec<Token>, Error = Simple<TokenKind>> {
    (just(TokenKind::Whitespace).repeated()).padding_for(
        any()
            .then(just(TokenKind::Whitespace).ignored().repeated())
            .map(|(kind, whitespace)| Token {
                kind,
                has_space_before: Some(!whitespace.is_empty()),
            })
            .repeated(),
    )
}

pub fn get_tokens(s: &str) -> Result<Vec<Token>, Vec<Error>> {
    let token_kinds = token_kind().repeated().padded_by(end()).parse(s)?;
    Ok(tokens()
        .padded_by(end())
        .parse(token_kinds)
        .expect("whitespace folding failed"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_tokens(input: &str, expected: &[TokenKind]) {
        let tokens = get_tokens(input).unwrap();
        let token_kinds: Vec<_> = tokens.into_iter().map(|token| token.kind).collect();
        assert_eq!(token_kinds, expected);
    }

    #[test]
    fn comment() {
        assert_tokens("// This is a line_comment", &[]);
    }

    #[test]
    fn block_comment() {
        assert_tokens("/* This is a block comment */", &[]);
    }

    #[test]
    fn block_comment_with_asterisk() {
        assert_tokens(
            "/** This *is* a block comment with embedded asterisks. **/",
            &[],
        );
    }

    #[test]
    fn string() {
        assert_tokens(
            r#" "Hello World" "#,
            &[TokenKind::String("Hello World".to_string())],
        );
    }

    #[test]
    fn string_with_escapes() {
        assert_tokens(
            r#" "Hello\r\n\    \t \"World!\"\x0A" "#,
            &[TokenKind::String("Hello\r\n\t \"World!\"\n".to_string())],
        );
    }

    #[test]
    fn keywords() {
        assert_tokens("true false", &[TokenKind::True, TokenKind::False])
    }

    #[test]
    fn float() {
        assert_tokens("2.0 1e6 2.99e8", &[TokenKind::Float(2.0), TokenKind::Float(1e6), TokenKind::Float(2.99e8)])
    }
}
