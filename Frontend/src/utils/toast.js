import toast from 'react-hot-toast';

// Enhanced toast functions with consistent styling
export const showToast = {
  success: (message, options = {}) => {
    return toast.success(message, {
      icon: '🎉',
      ...options,
    });
  },

  error: (message, options = {}) => {
    return toast.error(message, {
      icon: '❌',
      ...options,
    });
  },

  warning: (message, options = {}) => {
    return toast(message, {
      icon: '⚠️',
      duration: 4000,
      style: {
        border: '1px solid #f59e0b',
        background: 'linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%)',
        color: '#92400e',
      },
      ...options,
    });
  },

  info: (message, options = {}) => {
    return toast(message, {
      icon: 'ℹ️',
      duration: 4000,
      style: {
        border: '1px solid #3b82f6',
        background: 'linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)',
        color: '#1e40af',
      },
      ...options,
    });
  },

  loading: (message, options = {}) => {
    return toast.loading(message, {
      ...options,
    });
  },

  promise: (promise, messages, options = {}) => {
    return toast.promise(promise, messages, {
      loading: {
        icon: '⏳',
      },
      success: {
        icon: '✅',
      },
      error: {
        icon: '💥',
      },
      ...options,
    });
  },

  dismiss: (toastId) => {
    return toast.dismiss(toastId);
  },

  dismissAll: () => {
    return toast.dismiss();
  },
};

// Validation toast helpers
export const validationToast = {
  required: (field) => showToast.warning(`${field} is required`),
  invalid: (field) => showToast.warning(`Please enter a valid ${field}`),
  mismatch: (field1, field2) => showToast.error(`${field1} and ${field2} do not match`),
  minLength: (field, length) => showToast.warning(`${field} must be at least ${length} characters`),
  terms: () => showToast.warning('You must agree to the terms and conditions'),
};

// Auth-specific toast helpers
export const authToast = {
  loginSuccess: () => showToast.success('Login successful! Redirecting...', { duration: 3000 }),
  loginError: (message) => showToast.error(message || 'Login failed'),
  signupSuccess: () => showToast.success('Account created successfully!', { duration: 3000 }),
  signupError: (message) => showToast.error(message || 'Failed to create account'),
  authenticating: () => showToast.loading('Authenticating...'),
  creatingAccount: () => showToast.loading('Creating account...'),
  connectionError: () => showToast.error('Connection error. Please try again.', { duration: 6000 }),
};