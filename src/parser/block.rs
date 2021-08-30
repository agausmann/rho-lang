use chumsky::prelude::*;
use chumsky::stream::Stream;

use crate::parser::expr::Expr;
use crate::parser::token::{token, Token, TokenKind};

pub type Error = Simple<Token>;

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub statements: Vec<Statement>,
    pub value: Option<Box<Expr>>,
}

impl Block {
    fn invalid() -> Self {
        Self {
            statements: Vec::new(),
            value: Some(Box::new(Expr::Invalid)),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Expr(Expr),
}

/// Used by Statement parser to indicate whether it might be followed by another statement.
/// Usually `Continue`, but may be `Break` if an expression is parsed that does not end with a
/// semicolon. That indicates to a block parser that it should stop parsing, because that
/// expression should be the last one in the block (and is the value it evalutes to).
enum ParseControlFlow {
    /// May be followed by another statement.
    Continue,
    /// Must be the last statement.
    Break,
}

fn statement<E>(expr: E) -> impl Parser<Token, (Statement, ParseControlFlow), Error = Error>
where
    E: Parser<Token, Expr, Error = Error>,
{
    expr.map(Statement::Expr)
        .then(
            token(TokenKind::Semicolon)
                .or_not()
                .map(|option| match option {
                    Some(_) => ParseControlFlow::Continue,
                    None => ParseControlFlow::Break,
                }),
        )
}

/// Implemented as a procedural parse function instead of combinators,
/// because the control flow is easier to express this way.
struct BlockContents<E> {
    expr: E,
}

impl<E> Parser<Token, Block> for BlockContents<E>
where
    E: Clone + Parser<Token, Expr, Error = Error>,
{
    type Error = Simple<Token>;

    fn parse_inner<S>(
        &self,
        stream: &mut S,
        errors: &mut Vec<Self::Error>,
    ) -> (usize, Result<(Block, Option<Self::Error>), Self::Error>)
    where
        S: Stream<Token, <Self::Error as chumsky::Error<Token>>::Span>,
    {
        let mut statements = Vec::new();
        let mut value = None;
        let statement = statement(self.expr.clone());
        let mut consumed = 0;

        let mut errors_to_merge = Vec::new();

        loop {
            let (part, result) = statement.parse_inner(stream, errors);
            consumed += part;
            match result {
                Ok(((stmt, control_flow), maybe_error)) => {
                    errors_to_merge.extend(maybe_error);
                    match control_flow {
                        ParseControlFlow::Continue => {
                            statements.push(stmt);
                        }
                        ParseControlFlow::Break => {
                            match stmt {
                                Statement::Expr(expr) => {
                                    value = Some(Box::new(expr));
                                } // _ => unreachable!(),
                            }
                            break;
                        }
                    }
                }
                Err(e) => {
                    errors_to_merge.push(e);
                    if part == 0 {
                        // end of block
                        break;
                    } else {
                        return (consumed, Err(merge_all(errors_to_merge).unwrap()));
                    }
                }
            }
        }
        (
            consumed,
            Ok((Block { statements, value }, merge_all(errors_to_merge))),
        )
    }
}

fn merge_all(errors: Vec<Error>) -> Option<Error> {
    errors.into_iter().reduce(|a, b| a.merge(b))
}

pub fn block<E>(expr: E) -> impl Parser<Token, Block, Error = Error>
where
    E: Clone + Parser<Token, Expr, Error = Error>,
{
    BlockContents { expr }
        .delimited_by(
            Token::of(TokenKind::LeftBrace),
            Token::of(TokenKind::RightBrace),
        )
        .map(|option| option.unwrap_or(Block::invalid()))
}
