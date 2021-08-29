use crate::parser::expr::BinaryOp;
use crate::parser::expr::Expr;
use crate::parser::expr::UnaryOp;
use crate::value::Value;

#[derive(Debug)]
pub enum EvalError {
    Invalid,
    TypeError,
}

pub fn eval_expr(expr: &Expr) -> Result<Value, EvalError> {
    match expr {
        Expr::Invalid => Err(EvalError::Invalid),
        Expr::Literal(value) => Ok(value.clone()),
        Expr::Variable(_) => todo!(),
        Expr::UnaryOp(boxed) => {
            let (op, base_expr) = boxed.as_ref();
            let base = eval_expr(base_expr)?;
            match op {
                UnaryOp::Neg => base.neg(),
                UnaryOp::Not => base.not(),
                UnaryOp::Call(_) => todo!(),
                UnaryOp::Index(_) => todo!(),
                UnaryOp::Member(_) => todo!(),
            }
        }
        Expr::BinaryOp(boxed) => {
            let (op, left_expr, right_expr) = boxed.as_ref();
            let left = eval_expr(left_expr)?;
            if let Some(value) = op.short_circuit(&left)? {
                Ok(value)
            } else {
                let right = eval_expr(right_expr)?;
                match op {
                    BinaryOp::Add => left.add(right),
                    BinaryOp::Sub => left.sub(right),
                    BinaryOp::Mul => left.mul(right),
                    BinaryOp::Div => left.div(right),
                    BinaryOp::Rem => left.rem(right),
                    BinaryOp::And => left.and(right),
                    BinaryOp::Or => left.or(right),
                    BinaryOp::BitAnd => left.bitand(right),
                    BinaryOp::BitOr => left.bitor(right),
                    BinaryOp::BitXor => left.bitxor(right),
                    BinaryOp::Eq => Ok(Value::Bool(left == right)),
                    BinaryOp::Ne => Ok(Value::Bool(left != right)),
                    BinaryOp::Lt => left.lt(right),
                    BinaryOp::Le => left.le(right),
                    BinaryOp::Gt => left.gt(right),
                    BinaryOp::Ge => left.ge(right),
                    BinaryOp::Assign => todo!(),
                    BinaryOp::AddAssign => todo!(),
                    BinaryOp::SubAssign => todo!(),
                    BinaryOp::MulAssign => todo!(),
                    BinaryOp::DivAssign => todo!(),
                    BinaryOp::RemAssign => todo!(),
                    BinaryOp::AndAssign => todo!(),
                    BinaryOp::OrAssign => todo!(),
                    BinaryOp::BitAndAssign => todo!(),
                    BinaryOp::BitOrAssign => todo!(),
                    BinaryOp::BitXorAssign => todo!(),
                }
            }
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
        let value = eval_expr(&expr).unwrap();
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
}
