use std::cmp::Ordering;

use crate::eval::EvalError;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
}

impl Value {
    pub fn as_bool(&self) -> Result<bool, EvalError> {
        match self {
            &Self::Bool(x) => Ok(x),
            _ => Err(EvalError::TypeError),
        }
    }

    pub fn neg(self) -> Result<Value, EvalError> {
        match self {
            Self::Int(x) => Ok(Self::Int(-x)),
            Self::Float(x) => Ok(Self::Float(-x)),
            _ => Err(EvalError::TypeError),
        }
    }

    pub fn not(self) -> Result<Value, EvalError> {
        match self {
            Self::Bool(x) => Ok(Self::Bool(!x)),
            Self::Int(x) => Ok(Self::Int(!x)),
            _ => Err(EvalError::TypeError),
        }
    }

    pub fn add(self, right: Value) -> Result<Value, EvalError> {
        match (self, right) {
            (Self::Int(x), Self::Int(y)) => Ok(Self::Int(x + y)),
            (Self::Float(x), Self::Float(y)) => Ok(Self::Float(x + y)),
            _ => Err(EvalError::TypeError),
        }
    }

    pub fn sub(self, right: Value) -> Result<Value, EvalError> {
        match (self, right) {
            (Self::Int(x), Self::Int(y)) => Ok(Self::Int(x - y)),
            (Self::Float(x), Self::Float(y)) => Ok(Self::Float(x - y)),
            _ => Err(EvalError::TypeError),
        }
    }

    pub fn mul(self, right: Value) -> Result<Value, EvalError> {
        match (self, right) {
            (Self::Int(x), Self::Int(y)) => Ok(Self::Int(x * y)),
            (Self::Float(x), Self::Float(y)) => Ok(Self::Float(x * y)),
            _ => Err(EvalError::TypeError),
        }
    }

    pub fn div(self, right: Value) -> Result<Value, EvalError> {
        match (self, right) {
            (Self::Int(x), Self::Int(y)) => Ok(Self::Int(x / y)),
            (Self::Float(x), Self::Float(y)) => Ok(Self::Float(x / y)),
            _ => Err(EvalError::TypeError),
        }
    }

    pub fn rem(self, right: Value) -> Result<Value, EvalError> {
        match (self, right) {
            (Self::Int(x), Self::Int(y)) => Ok(Self::Int(x % y)),
            (Self::Float(x), Self::Float(y)) => Ok(Self::Float(x % y)),
            _ => Err(EvalError::TypeError),
        }
    }

    pub fn and(self, right: Value) -> Result<Value, EvalError> {
        match (self, right) {
            (Self::Bool(x), Self::Bool(y)) => Ok(Self::Bool(x && y)),
            _ => Err(EvalError::TypeError),
        }
    }

    pub fn or(self, right: Value) -> Result<Value, EvalError> {
        match (self, right) {
            (Self::Bool(x), Self::Bool(y)) => Ok(Self::Bool(x || y)),
            _ => Err(EvalError::TypeError),
        }
    }

    pub fn bitand(self, right: Value) -> Result<Value, EvalError> {
        match (self, right) {
            (Self::Bool(x), Self::Bool(y)) => Ok(Self::Bool(x & y)),
            (Self::Int(x), Self::Int(y)) => Ok(Self::Int(x & y)),
            _ => Err(EvalError::TypeError),
        }
    }

    pub fn bitor(self, right: Value) -> Result<Value, EvalError> {
        match (self, right) {
            (Self::Bool(x), Self::Bool(y)) => Ok(Self::Bool(x | y)),
            (Self::Int(x), Self::Int(y)) => Ok(Self::Int(x | y)),
            _ => Err(EvalError::TypeError),
        }
    }

    pub fn bitxor(self, right: Value) -> Result<Value, EvalError> {
        match (self, right) {
            (Self::Bool(x), Self::Bool(y)) => Ok(Self::Bool(x ^ y)),
            (Self::Int(x), Self::Int(y)) => Ok(Self::Int(x ^ y)),
            _ => Err(EvalError::TypeError),
        }
    }

    pub fn cmp(self, right: Value) -> Result<Option<Ordering>, EvalError> {
        match (self, right) {
            (Self::Bool(x), Self::Bool(y)) => Ok(x.partial_cmp(&y)),
            (Self::Int(x), Self::Int(y)) => Ok(x.partial_cmp(&y)),
            (Self::Float(x), Self::Float(y)) => Ok(x.partial_cmp(&y)),
            _ => Err(EvalError::TypeError)
        }
    }

    pub fn lt(self, right: Value) -> Result<Value, EvalError> {
        self.cmp(right).map(|ordering| Value::Bool(ordering.map(Ordering::is_lt).unwrap_or(false)))
    }

    pub fn le(self, right: Value) -> Result<Value, EvalError> {
        self.cmp(right).map(|ordering| Value::Bool(ordering.map(Ordering::is_le).unwrap_or(false)))
    }

    pub fn gt(self, right: Value) -> Result<Value, EvalError> {
        self.cmp(right).map(|ordering| Value::Bool(ordering.map(Ordering::is_gt).unwrap_or(false)))
    }

    pub fn ge(self, right: Value) -> Result<Value, EvalError> {
        self.cmp(right).map(|ordering| Value::Bool(ordering.map(Ordering::is_ge).unwrap_or(false)))
    }
}
