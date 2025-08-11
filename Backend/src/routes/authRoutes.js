import express from 'express';
import { 
  signup, 
  login, 
  logout, 
  getProfile,
  updateProfile,
  updatePassword,
  updateAvatar,
  deleteAvatar,
  getFullProfile
} from '../controllers/authController.js';
import { validateSignup, validateLogin } from '../middleware/validation.js';
import { authenticateToken } from '../middleware/auth.js';

const router = express.Router();

// Public routes
router.post('/signup', validateSignup, signup);
router.post('/login', validateLogin, login);
router.post('/logout', logout);
router.get('/login', (req, res) => {
  res.status(200).json({
    success: true,
    message: 'Login route is working. Use POST /api/auth/login to authenticate.'
  });
});
router.get('/signup', (req, res) => {
  res.status(200).json({
    success: true,
    message: 'Signup route is working. Use POST /api/auth/signup to create an account.'
  });
});

// Protected routes - Basic profile
router.get('/profile', authenticateToken, getProfile);
router.get('/profile/full', authenticateToken, getFullProfile);

// Protected routes - Profile management
router.put('/profile', authenticateToken, updateProfile);
router.put('/profile/password', authenticateToken, updatePassword);
router.put('/profile/avatar', authenticateToken, updateAvatar);
router.delete('/profile/avatar', authenticateToken, deleteAvatar);

export default router;