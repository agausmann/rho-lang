use std::collections::HashMap;
use std::mem::replace;

use crate::parser::block::Statement;
use crate::parser::expr::{BinaryOp, Expr, UnaryOp};
use crate::value::Value;

#[derive(Debug)]
pub enum EvalError {
    Invalid,
    TypeError,
    NameError,
    SyntaxError,
}

pub struct Scope {
    variables: HashMap<String, Value>,
}

impl Scope {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    pub fn get(&self, key: &str) -> Result<Value, EvalError> {
        self.variables.get(key).cloned().ok_or(EvalError::NameError)
    }

    pub fn get_mut(&mut self, key: &str) -> Result<&mut Value, EvalError> {
        self.variables.get_mut(key).ok_or(EvalError::NameError)
    }

    pub fn get_or_insert(&mut self, key: &str, value: Value) -> &mut Value {
        self.variables.entry(key.to_string()).or_insert(value)
    }
}

struct Lvalue<'a> {
    slot: &'a mut Value,
}

impl<'a> Lvalue<'a> {
    fn set(&mut self, value: Value) -> Result<Value, EvalError> {
        *self.slot = value;
        Ok(Value::Null)
    }

    fn try_update<F>(&mut self, f: F) -> Result<Value, EvalError>
    where
        F: FnOnce(Value) -> Result<Value, EvalError>,
    {
        // swap with a placeholder value so we can consume the original value
        let value = replace(self.slot, Value::Null);
        *self.slot = f(value)?;
        Ok(Value::Null)
    }
}

fn lvalue<'a>(scope: &'a mut Scope, expr: &Expr) -> Result<Lvalue<'a>, EvalError> {
    let slot = match expr {
        Expr::Variable(name) => {
            // XXX insert should be deferred until after RHS is evaluated,
            // and only for `set`, not `update`.
            scope.get_or_insert(&name, Value::Null)
        }
        _ => return Err(EvalError::SyntaxError),
    };
    Ok(Lvalue { slot })
}

pub fn eval(scope: &mut Scope, expr: &Expr) -> Result<Value, EvalError> {
    match expr {
        Expr::Invalid => Err(EvalError::Invalid),
        Expr::Literal(value) => Ok(value.clone()),
        Expr::Variable(name) => scope.get(&name),
        Expr::UnaryOp(boxed) => {
            let (op, base_expr) = boxed.as_ref();
            let base = eval(scope, base_expr)?;
            match op {
                UnaryOp::Neg => base.neg(),
                UnaryOp::Not => base.not(),
                UnaryOp::Call(_) => todo!(),
                UnaryOp::Index(_) => todo!(),
                UnaryOp::Member(_) => todo!(),
            }
        }
        Expr::BinaryOp(boxed) => {
            let (op, left, right) = boxed.as_ref();
            match op {
                BinaryOp::Add => eval(scope, left)?.add(eval(scope, right)?),
                BinaryOp::Sub => eval(scope, left)?.sub(eval(scope, right)?),
                BinaryOp::Mul => eval(scope, left)?.mul(eval(scope, right)?),
                BinaryOp::Div => eval(scope, left)?.div(eval(scope, right)?),
                BinaryOp::Rem => eval(scope, left)?.rem(eval(scope, right)?),
                BinaryOp::And => Ok(Value::Bool(
                    eval(scope, left)?.as_bool()? && eval(scope, right)?.as_bool()?,
                )),
                BinaryOp::Or => Ok(Value::Bool(
                    eval(scope, left)?.as_bool()? || eval(scope, right)?.as_bool()?,
                )),
                BinaryOp::BitAnd => eval(scope, left)?.bitand(eval(scope, right)?),
                BinaryOp::BitOr => eval(scope, left)?.bitor(eval(scope, right)?),
                BinaryOp::BitXor => eval(scope, left)?.bitxor(eval(scope, right)?),
                BinaryOp::Eq => Ok(Value::Bool(eval(scope, left)? == eval(scope, right)?)),
                BinaryOp::Ne => Ok(Value::Bool(eval(scope, left)? != eval(scope, right)?)),
                BinaryOp::Lt => eval(scope, left)?.lt(eval(scope, right)?),
                BinaryOp::Le => eval(scope, left)?.le(eval(scope, right)?),
                BinaryOp::Gt => eval(scope, left)?.gt(eval(scope, right)?),
                BinaryOp::Ge => eval(scope, left)?.ge(eval(scope, right)?),
                BinaryOp::Assign => {
                    //TODO is this evaluation order correct?
                    let right = eval(scope, right)?;
                    lvalue(scope, left)?.set(right)
                }
                BinaryOp::AddAssign => {
                    let right = eval(scope, right)?;
                    lvalue(scope, left)?.try_update(|left| left.add(right))
                }
                BinaryOp::SubAssign => {
                    let right = eval(scope, right)?;
                    lvalue(scope, left)?.try_update(|left| left.sub(right))
                }
                BinaryOp::MulAssign => {
                    let right = eval(scope, right)?;
                    lvalue(scope, left)?.try_update(|left| left.mul(right))
                }
                BinaryOp::DivAssign => {
                    let right = eval(scope, right)?;
                    lvalue(scope, left)?.try_update(|left| left.div(right))
                }
                BinaryOp::RemAssign => {
                    let right = eval(scope, right)?;
                    lvalue(scope, left)?.try_update(|left| left.rem(right))
                }
                BinaryOp::AndAssign => {
                    let left_value = eval(scope, left)?;
                    let right_value = eval(scope, right)?;
                    lvalue(scope, left)?
                        .set(Value::Bool(left_value.as_bool()? && right_value.as_bool()?))
                }
                BinaryOp::OrAssign => {
                    let left_value = eval(scope, left)?;
                    let right_value = eval(scope, right)?;
                    lvalue(scope, left)?
                        .set(Value::Bool(left_value.as_bool()? && right_value.as_bool()?))
                }
                BinaryOp::BitAndAssign => {
                    let right = eval(scope, right)?;
                    lvalue(scope, left)?.try_update(|left| left.bitand(right))
                }
                BinaryOp::BitOrAssign => {
                    let right = eval(scope, right)?;
                    lvalue(scope, left)?.try_update(|left| left.bitor(right))
                }
                BinaryOp::BitXorAssign => {
                    let right = eval(scope, right)?;
                    lvalue(scope, left)?.try_update(|left| left.bitxor(right))
                }
            }
        }
        Expr::Block(block) => {
            for statement in &block.statements {
                exec(scope, statement)?;
            }
            if let Some(value_expr) = &block.value {
                eval(scope, value_expr)
            } else {
                Ok(Value::Null)
            }
        }
    }
}

pub fn exec(scope: &mut Scope, statement: &Statement) -> Result<(), EvalError> {
    match statement {
        Statement::Expr(expr) => {
            eval(scope, expr)?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::parser::expr::expr;
    use crate::parser::token::get_tokens;
    use chumsky::prelude::*;

    use super::*;

    fn assert_eval(expr_str: &str, expected: Value) {
        let tokens = get_tokens(expr_str).unwrap();
        let expr = expr().padded_by(end()).parse(tokens).unwrap();
        let value = eval(&mut Scope::new(), &expr).unwrap();
        assert_eq!(expected, value);
    }

    #[test]
    fn two_plus_two() {
        assert_eval("2+2", Value::Int(2 + 2))
    }

    #[test]
    fn complex_arithmetic() {
        assert_eval(
            "(1 + 2) + (3 * -(4 * 5 - 6))",
            Value::Int((1 + 2) + (3 * -(4 * 5 - 6))),
        )
    }

    #[test]
    fn two_plus_two_is_four() {
        assert_eval("2 + 2 == 4", Value::Bool(true));
    }

    #[test]
    fn two_plus_two_not_five() {
        assert_eval("2 + 2 == 5", Value::Bool(false));
    }

    #[test]
    fn block() {
        assert_eval("{ a = 2; a }", Value::Int(2));
    }

    #[test]
    fn block_assign_multiple() {
        assert_eval("{ foo = 0; foo += 1; foo -= 10; foo + 5 }", Value::Int(0 + 1 - 10 + 5));
    }
}
