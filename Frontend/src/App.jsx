import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

// Layout Components
import Layout from "./components/layout/Layout";
import AuthLayout from "./components/layout/AuthLayout";

// User Profile Component
import Profile from "./components/user/profile";

// Pages
import Home from "./pages/Home";
import Setup from "./pages/Setup";
import Authentication from "./pages/Authentication";
import { Dashboard } from "./pages/Dashboard";

// Route Guards
import ProtectedRoute from "./components/ProtectedRoute";
import PublicRoute from "./components/PublicRoute";

// Toast Provider
import ToastProvider from "./providers/ToastProvider";
import Error from "./components/Error";
import { UploadPage } from "./pages/Upload";

function App() {
    return (
        <Router>
            <Routes>
                {/* Public Routes with Main Layout */}
                <Route
                    path="/"
                    element={
                        <Layout>
                            <Home />
                        </Layout>
                    }
                />

                <Route
                    path="/setup"
                    element={
                        <Layout>
                            <Setup />
                        </Layout>
                    }
                />

                {/* Auth Routes with Auth Layout - Redirect if already logged in */}
                <Route
                    path="/auth"
                    element={
                        <PublicRoute>
                            <AuthLayout>
                                <Authentication />
                            </AuthLayout>
                        </PublicRoute>
                    }
                />

                {/* Dashboard Route - Any authenticated user */}
                <Route
                    path="/dashboard"
                    element={
                        <ProtectedRoute>
                            <Layout>
                                <Dashboard />
                            </Layout>
                        </ProtectedRoute>
                    }
                />

                {/* Protected Routes with Role Requirements */}
                <Route
                    path="/admin"
                    element={
                        <ProtectedRoute roles={["ADMIN"]}>
                            <Layout>
                                <div className="p-8">
                                    <h1 className="text-2xl font-bold">
                                        Admin Dashboard
                                    </h1>
                                    <p>This page is only for administrators.</p>
                                </div>
                            </Layout>
                        </ProtectedRoute>
                    }
                />

                {/* Profile Route - Any authenticated user */}
                <Route
                    path="/profile"
                    element={
                        <ProtectedRoute>
                            <Layout>
                                <Profile />
                            </Layout>
                        </ProtectedRoute>
                    }
                />

                {/* Upload Route - Any authenticated user */}
                <Route
                    path="/upload"
                    element={
                        <ProtectedRoute>
                            <Layout>
                                <UploadPage />
                            </Layout>
                        </ProtectedRoute>
                    }
                />

                <Route
                    path="*"
                    element={
                        <Layout>
                            <Error />
                        </Layout>
                    }
                />
            </Routes>

            <ToastProvider />
        </Router>
    );
}

export default App;
