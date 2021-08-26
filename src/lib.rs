use chumsky::prelude::*;

const WHITESPACE: &str = " \t\n\r";
const ALPHA_UND: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_";
const ALPHA_NUM_UND: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_";
const NONZERO_DIGIT: &str = "123456789";
const DIGIT: &str = "0123456789";
const HEXDIGIT: &str = "0123456789ABCDEFabcdef";
const BINDIGIT: &str = "01";

#[derive(Debug, Clone)]
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

fn nonzero_digit() -> impl Parser<char, char, Error = Simple<char>> {
    one_of(NONZERO_DIGIT.chars())
}

fn digit() -> impl Parser<char, char, Error = Simple<char>> {
    one_of(DIGIT.chars())
}

fn hexdigit() -> impl Parser<char, char, Error = Simple<char>> {
    one_of(HEXDIGIT.chars())
}

fn bindigit() -> impl Parser<char, char, Error = Simple<char>> {
    one_of(BINDIGIT.chars())
}

fn float_exponent() -> impl Parser<char, Vec<char>, Error = Simple<char>> {
    one_of("eE".chars())
        .chain(one_of("+-".chars()).or_not())
        .chain(digit().repeated_at_least(1))
}

fn float_after_decimal() -> impl Parser<char, Vec<char>, Error = Simple<char>> {
    digit().repeated_at_least(1)
}

fn float_int_suffix() -> impl Parser<char, Vec<char>, Error = Simple<char>> {
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
fn leading_digit_number() -> impl Parser<char, Token, Error = Simple<char>> {
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

fn string_contents() -> impl Parser<char, Token, Error = Simple<char>> {
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
                .or(one_of(WHITESPACE.chars()).repeated_at_least(1).to(None))
        ))
        .repeated()
        .flatten()
        .collect()
        .map(Token::String)
}

fn tokenizer() -> impl Parser<char, Token, Error = Simple<char>> {
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

pub fn get_tokens(s: &str) -> Result<Vec<Token>, Vec<Simple<char>>> {
    tokenizer().repeated().padded_by(end()).parse(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_parse(input: &str) {
        dbg!(get_tokens(input).unwrap());
    }

    #[test]
    fn comment() {
        assert_parse("// This is a line_comment");
    }

    #[test]
    fn block_comment() {
        assert_parse("/* This is a block comment */");
    }

    #[test]
    fn block_comment_with_asterisk() {
        assert_parse("/** This *is* a block comment with embedded asterisks. **/");
    }

    #[test]
    fn string() {
        assert_parse(r#" "Hello World" "#);
    }

    #[test]
    fn string_with_escapes() {
        assert_parse(r#" "Hello\r\n\    \t \"World!\"\x0A" "#);
    }
}