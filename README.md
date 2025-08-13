# Drishtiksha 🚀

A modern full-stack web application starter template built with **React**, **Node.js**, **Express**, **Prisma**, and **Tailwind CSS**. This starter pack provides a complete development environment with authentication, database integration, and a beautiful dark/light theme system.

## ✨ Features

- 🎨 **Modern UI/UX** - Beautiful responsive design with dark/light theme
- ⚡ **Fast Development** - Hot reload with Vite for frontend and Nodemon for backend
- 🔐 **Authentication System** - JWT-based user authentication
- 🗃️ **Database Integration** - Prisma ORM with SQLite (easily configurable for other databases)
- 📱 **Responsive Design** - Mobile-first approach with Tailwind CSS
- 🛠️ **Interactive Setup Guide** - Built-in setup wizard for easy project initialization
- 🔧 **Developer Experience** - ESLint, PostCSS, and modern tooling
- 🌙 **Theme System** - Seamless dark/light mode switching
- 📊 **Health Monitoring** - Backend health check endpoints

## 🛠️ Tech Stack

### Frontend
- **React 19** - Latest React with modern features
- **Vite** - Fast build tool and development server
- **Tailwind CSS** - Utility-first CSS framework
- **React Router** - Client-side routing
- **Lucide React** - Beautiful icon library
- **React Hot Toast** - Elegant toast notifications
- **Framer Motion** - Smooth animations

### Backend
- **Node.js** - JavaScript runtime
- **Express.js** - Web application framework
- **Prisma ORM** - Modern database toolkit
- **JWT** - JSON Web Token authentication
- **bcryptjs** - Password hashing
- **CORS** - Cross-origin resource sharing
- **SQLite** - Default database (configurable)

## 📁 Project Structure

```
├── Frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── layout/
│   │   │   │   ├── Header.jsx
│   │   │   │   └── Footer.jsx
│   │   │   └── ThemeToggle.jsx
│   │   ├── pages/
│   │   │   ├── Home.jsx
│   │   │   └── Setup.jsx
│   │   ├── providers/
│   │   ├── utils/
│   │   └── assets/
│   ├── public/
│   ├── package.json
│   ├── tailwind.config.js
│   ├── vite.config.js
│   └── .env
│
├── Backend/
│   ├── src/
│   │   ├── controllers/
│   │   │   └── authController.js
│   │   ├── middleware/
│   │   │   ├── auth.js
│   │   │   └── validation.js
│   │   ├── routes/
│   │   │   └── authRoutes.js
│   │   ├── config/
│   │   │   └── database.js
│   │   └── utils/
│   ├── prisma/
│   │   └── schema.prisma
│   ├── package.json
│   ├── server.js
│   └── .env
│
└── README.md
```

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have the following installed:
- **Node.js** (v18.0.0 or higher) - [Download](https://nodejs.org/)
- **npm** (v8.0.0 or higher) - Comes with Node.js
- **Git** (Latest version) - [Download](https://git-scm.com/)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/slantie/StarterKit.git
   cd starter-pack
   ```

2. **Frontend Setup**
   ```bash
   cd Frontend
   npm install
   cp .env.example .env  # Create environment file
   npm run dev           # Start development server on :5173
   ```

3. **Backend Setup** (Open a new terminal)
   ```bash
   cd Backend
   npm install
   cp .env.example .env  # Create environment file
   npm run dev           # Start backend server on :3000
   ```

4. **Database Setup** (In the Backend directory)
   ```bash
   npm run db:generate   # Generate Prisma client
   npm run db:push       # Create database tables
   npm run db:studio     # (Optional) Open Prisma Studio
   ```

5. **Verify Installation**
   - Frontend: [http://localhost:5173](http://localhost:5173)
   - Backend Health: [http://localhost:3000/api/health](http://localhost:3000/api/health)
   - Interactive Setup Guide: [http://localhost:5173/setup](http://localhost:5173/setup)

## ⚙️ Configuration

### Environment Variables

#### Frontend (.env)
```env
# Project Configuration
VITE_PROJECT_NAME="Your Project Name"
VITE_BACKEND_URL="http://localhost:3000"
VITE_NODE_ENV="development"

# Development Server
PORT=5173
NODE_ENV="development"
```

#### Backend (.env)
```env
# Server Configuration
PORT=3000
NODE_ENV="development"

# Database
DATABASE_URL="file:./dev.db"

# Authentication
JWT_SECRET="your-super-secret-jwt-key-here"

# CORS
FRONTEND_URL="http://localhost:5173"
```

### Database Configuration

The project uses SQLite by default, but you can easily switch to other databases by updating the `DATABASE_URL` in your Backend `.env` file:

```env
# PostgreSQL
DATABASE_URL="postgresql://user:password@localhost:5432/dbname"

# MySQL
DATABASE_URL="mysql://user:password@localhost:3306/dbname"

# MongoDB
DATABASE_URL="mongodb://localhost:27017/dbname"
```

## 📋 Available Scripts

### Frontend Scripts
```bash
npm run dev         # Start development server
npm run build       # Build for production
npm run preview     # Preview production build
npm run lint        # Run ESLint
```

### Backend Scripts
```bash
npm run dev         # Start development server with nodemon
npm start           # Start production server
npm run db:generate # Generate Prisma client
npm run db:push     # Push schema to database
npm run db:studio   # Open Prisma Studio
```

## 🎨 Customization

### Theme Configuration

The project includes a comprehensive theme system defined in [`tailwind.config.js`](Frontend/tailwind.config.js):

```javascript
// Custom color palette
colors: {
  primary: {
    main: "#f56565",
    light: "#ff7070",
    // ... more variants
  },
  dark: {
    text: "#DFDFD6",
    background: "#1B1B1F",
    // ... dark theme colors
  },
  light: {
    text: "#3C3C43",
    background: "#FFFFFF",
    // ... light theme colors
  }
}
```

### Adding New Pages

1. Create a new component in [`Frontend/src/pages/`](Frontend/src/pages/)
2. Add routing in your main App component
3. Update navigation in [`Header.jsx`](Frontend/src/components/layout/Header.jsx)

## 📱 Features Overview

### Interactive Setup Guide
The project includes a comprehensive setup guide accessible at `/setup` that provides:
- Step-by-step installation instructions
- Environment variable configuration
- Health check monitoring
- Command copying functionality
- Progress tracking

### Authentication System
- JWT-based authentication
- Password hashing with bcryptjs
- Protected routes middleware
- User session management

### Theme System
- Automatic dark/light mode detection
- Manual theme switching
- Consistent color palette
- Responsive design patterns

## 🔧 Development

### Code Style
- ESLint configuration included
- Consistent formatting with Prettier (recommended)
- Modern JavaScript/React patterns
- Modular component architecture

### Git Workflow
The project includes comprehensive `.gitignore` files for both frontend and backend:
- Node modules exclusion
- Environment variables protection
- Build artifacts ignored
- Development files excluded

## 🚀 Deployment

### Frontend Deployment (Vercel/Netlify)
```bash
npm run build
# Deploy the `dist` folder
```

### Backend Deployment (Railway/Heroku)
```bash
# Update DATABASE_URL for production database
# Set JWT_SECRET to a secure random string
# Configure environment variables in your hosting platform
```

## 🔗 Links

- [Frontend Development Server](http://localhost:5173)
- [Backend API Health Check](http://localhost:3000/api/health)
- [Interactive Setup Guide](http://localhost:5173/setup)

---

**Happy Coding!** 🎉

Built with ❤️ using modern web technologies
