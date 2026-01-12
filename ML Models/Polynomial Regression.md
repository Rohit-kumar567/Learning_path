# Polynomial Regression

## Overview

Polynomial regression is a form of regression analysis where the relationship between the independent variable (x) and dependent variable (y) is modeled as an nth degree polynomial. It extends simple linear regression to capture non-linear relationships in data.

## Table of Contents

- [What is Polynomial Regression?](#what-is-polynomial-regression)
- [When to Use](#when-to-use-polynomial-regression)
- [Common Polynomial Degrees](#common-polynomial-degrees)
- [Advantages](#advantages)
- [Disadvantages](#disadvantages)
- [Implementation Steps](#implementation-steps)
- [Example Use Cases](#example-use-cases)
- [Key Considerations](#key-considerations)
- [Evaluation Metrics](#evaluation-metrics)
- [Alternatives](#alternatives-to-consider)
- [Getting Started](#getting-started)

## What is Polynomial Regression?

While linear regression fits a straight line to data, polynomial regression fits a curved line by adding polynomial terms. The general form is:

```
y = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ + ε
```

**Where:**
- `y` is the dependent variable
- `x` is the independent variable
- `β₀, β₁, ..., βₙ` are the coefficients
- `n` is the degree of the polynomial
- `ε` is the error term

## When to Use Polynomial Regression

Polynomial regression is appropriate when:

- The relationship between variables is curved rather than linear
- A scatterplot shows a non-linear pattern
- Linear regression produces poor results with high residual errors
- You need to model phenomena with increasing/decreasing rates of change

## Common Polynomial Degrees

| Degree | Name | Description |
|--------|------|-------------|
| 1 | Linear | Straight line |
| 2 | Quadratic | Parabola - captures one curve/bend |
| 3 | Cubic | Captures two curves/bends |
| 4+ | Higher-order | More complex patterns (risk of overfitting) |

## Advantages

✅ Can model non-linear relationships effectively  
✅ Still uses linear regression techniques (it's linear in the coefficients)  
✅ Flexible and can fit various curve shapes  
✅ Easy to implement with existing linear regression tools

## Disadvantages

❌ **Overfitting risk**: Higher degree polynomials can fit training data too closely  
❌ **Sensitive to outliers**: Extreme values can distort the curve significantly  
❌ **Extrapolation issues**: Predictions outside the data range can be unreliable  
❌ **Multicollinearity**: Higher polynomial terms are often highly correlated

## Implementation Steps

1. **Explore the data**: Create scatterplots to identify non-linear patterns
2. **Choose polynomial degree**: Start with degree 2 or 3, use cross-validation to optimize
3. **Transform features**: Create polynomial features (x², x³, etc.)
4. **Fit the model**: Use standard linear regression on transformed features
5. **Evaluate**: Check R², residual plots, and test set performance
6. **Validate**: Ensure the model doesn't overfit using cross-validation

## Example Use Cases

- 📈 Modeling population growth over time
- 🌾 Predicting crop yield based on fertilizer amount
- 🧪 Analyzing the relationship between temperature and chemical reaction rates
- 🚗 Modeling stopping distance of vehicles at different speeds
- 💰 Economic models with diminishing returns

## Key Considerations

### Feature Scaling
Polynomial features can have very different scales, so standardization is often necessary.

### Degree Selection
Use techniques like cross-validation, AIC, or BIC to choose the optimal degree. Start simple and increase complexity only if justified.

### Regularization
Consider Ridge or Lasso regression to prevent overfitting with higher-degree polynomials.

## Evaluation Metrics

- **R² Score**: Measures proportion of variance explained
- **Mean Squared Error (MSE)**: Average squared difference between predictions and actual values
- **Cross-validation score**: Tests generalization to unseen data
- **Residual plots**: Should show random scatter without patterns

## Alternatives to Consider

If polynomial regression doesn't work well, consider:

- Spline regression for more flexible curves
- Generalized additive models (GAMs)
- Non-parametric methods like LOESS
- Tree-based models for complex non-linear relationships

## Getting Started

Most statistical and machine learning libraries support polynomial regression through feature transformation combined with linear regression:

- **Python**: scikit-learn's `PolynomialFeatures`
- **R**: `lm()` function with `poly()`
- **MATLAB**: `polyfit()` function

### Quick Example (Python)

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Create polynomial regression model
degree = 2
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Fit and predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

## 📝 Note

> **Remember**: Polynomial regression is a powerful tool for non-linear relationships, but simpler models are often better. Always validate your model and be cautious of overfitting!

## 📚 Further Reading

- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression)
- [Statistical Learning Textbook](https://www.statlearning.com/)
- Cross-validation techniques for model selection

## 🤝 Contributing

Feel free to contribute to this documentation by submitting pull requests or opening issues.

## 📄 License

This documentation is provided as-is for educational purposes.

---

**Last Updated**: January 2026
