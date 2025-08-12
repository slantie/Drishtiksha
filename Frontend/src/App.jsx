// src/App.jsx

import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { AuthProvider } from "./contexts/AuthContext.jsx";

// Layout and Page Components
import Layout from "./components/layout/Layout";
import AuthLayout from "./components/layout/AuthLayout";
import ProtectedRoute from "./components/ProtectedRoute";
import PublicRoute from "./components/PublicRoute";
import ToastProvider from "./providers/ToastProvider";
import Error from "./components/Error";
import Home from "./pages/Home";
import Authentication from "./pages/Authentication";
import Profile from "./pages/Profile.jsx";
import Dashboard from "./pages/Dashboard";
import Results from "./pages/Results";

function App() {
    return (
        <Router>
            <AuthProvider>
                <Routes>
                    <Route
                        path="/"
                        element={
                            <Layout>
                                <Home />
                            </Layout>
                        }
                    />
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
                    <Route
                        path="/results/:videoId"
                        element={
                            <ProtectedRoute>
                                <Layout>
                                    <Results />
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
            </AuthProvider>
        </Router>
    );
}

export default App;
