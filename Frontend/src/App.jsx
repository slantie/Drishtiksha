// src/App.jsx

import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { AuthProvider } from "./contexts/AuthContext.jsx";
import { ThemeProvider } from "./contexts/ThemeContext.jsx";
import Layout from "./components/layout/Layout";
import AuthLayout from "./components/layout/AuthLayout";
import ProtectedRoute from "./components/ProtectedRoute";
import PublicRoute from "./components/PublicRoute";
import ToastProvider from "./providers/ToastProvider";
import { useToastOrchestrator } from "./lib/toastOrchestrator.jsx";
import Error from "./components/Error";
import Home from "./pages/Home.jsx";
import Authentication from "./pages/Authentication";
import Profile from "./pages/Profile.jsx";
import Dashboard from "./pages/Dashboard";
import Results from "./pages/Results";
import DetailedAnalysisPage from "./pages/Analysis.jsx";
import RunDetailsPage from "./pages/RunDetails.jsx";
import Monitoring from "./pages/Monitoring.jsx";
import Docs from "./pages/Docs.jsx";

function App() {
  // Initialize the toast orchestrator
  useToastOrchestrator();

  return (
    <Router>
      <ThemeProvider>
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
              path="/results/:mediaId"
              element={
                <ProtectedRoute>
                  <Layout>
                    <Results />
                  </Layout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/results/:mediaId/runs/:runId"
              element={
                <ProtectedRoute>
                  <Layout>
                    <RunDetailsPage />
                  </Layout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/results/:mediaId/:analysisId"
              element={
                <ProtectedRoute>
                  <Layout>
                    <DetailedAnalysisPage />
                  </Layout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/monitor"
              element={
                <ProtectedRoute>
                  <Layout>
                    <Monitoring />
                  </Layout>
                </ProtectedRoute>
              }
            />
            <Route
              path="/docs/*"
              element={
                <Layout>
                  <Docs />
                </Layout>
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
      </ThemeProvider>
    </Router>
  );
}

export default App;
