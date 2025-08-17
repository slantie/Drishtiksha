// src/middleware/security.middleware.js

/**
 * Security middleware to add security headers and protect against common attacks
 */
export const securityHeaders = (req, res, next) => {
    // Prevent clickjacking
    res.setHeader("X-Frame-Options", "DENY");

    // Prevent MIME sniffing
    res.setHeader("X-Content-Type-Options", "nosniff");

    // Enable XSS Protection
    res.setHeader("X-XSS-Protection", "1; mode=block");

    // Strict Transport Security (HTTPS only in production)
    if (process.env.NODE_ENV === "production") {
        res.setHeader(
            "Strict-Transport-Security",
            "max-age=31536000; includeSubDomains"
        );
    }

    // Content Security Policy (adjust as needed)
    res.setHeader("Content-Security-Policy", "default-src 'self'");

    // Referrer Policy
    res.setHeader("Referrer-Policy", "strict-origin-when-cross-origin");

    next();
};

/**
 * Middleware to validate secure token storage practices
 */
export const validateTokenSecurity = (req, res, next) => {
    const token =
        req.cookies?.authToken ||
        req.header("Authorization")?.replace("Bearer ", "");

    if (token) {
        // Log potential security issues (remove in production)
        if (process.env.NODE_ENV === "development") {
            const hasHttpOnlyCookie = !!req.cookies?.authToken;
            const hasAuthHeader = !!req.header("Authorization");

            console.log("Token Security Check:", {
                hasHttpOnlyCookie,
                hasAuthHeader,
                userAgent: req.get("User-Agent"),
                ip: req.ip,
            });
        }
    }

    next();
};
