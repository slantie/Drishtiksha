export const validateSignup = (req, res, next) => {
  const { email, firstName, lastName, password } = req.body;
  
  // Check required fields
  if (!email || !firstName || !lastName || !password) {
    return res.status(400).json({
      success: false,
      message: 'All fields are required'
    });
  }
  
  // Email validation
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email)) {
    return res.status(400).json({
      success: false,
      message: 'Please provide a valid email address'
    });
  }
  
  // Password validation
  if (password.length < 6) {
    return res.status(400).json({
      success: false,
      message: 'Password must be at least 6 characters long'
    });
  }
  
  // Name validation
  if (firstName.trim().length < 2 || lastName.trim().length < 2) {
    return res.status(400).json({
      success: false,
      message: 'First name and last name must be at least 2 characters long'
    });
  }
  
  next();
};

export const validateLogin = (req, res, next) => {
  const { email, password } = req.body;
  
  if (!email || !password) {
    return res.status(400).json({
      success: false,
      message: 'Email and password are required'
    });
  }
  
  next();
};