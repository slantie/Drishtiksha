// src/components/layout/Footer.jsx

import React from "react";
import { Github, Linkedin, ArrowUp } from "lucide-react";
import { Button } from "../ui/Button";
import { config } from "../../config/env.js";

const projectName = config.VITE_PROJECT_NAME || "Drishtiksha";

// Define social links. Could be moved to a global config or fetched if dynamic.
const socialLinks = [
  {
    name: "GitHub",
    icon: Github,
    url: "https://github.com/zaptrixio-cyber/Drishtiksha",
  },
];

function Footer() {
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <footer className="bg-light-background dark:bg-dark-muted-background border-t border-light-secondary dark:border-dark-secondary">
      <div className="mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col md:flex-row justify-between items-center gap-6">
          {/* Copyright Info */}
          <div className="text-sm text-light-muted-text dark:text-dark-muted-text">
            &copy; {new Date().getFullYear()} {projectName}. All Rights
            Reserved.
          </div>

          <div className="flex items-center gap-2 justify-center">
            {/* Social Links */}
            {/* <div className="flex items-center space-x-4">
              {socialLinks.map((link) => (
                <Button
                  as="a"
                  variant="outline"
                  key={link.name}
                  href={link.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-light-muted-text gap-2 dark:text-dark-muted-text hover:text-primary-main transition-colors"
                >
                  <link.icon className="h-5 w-5" />
                  <span>{link.name}</span>
                </Button>
              ))}
            </div> */}
            {/* Back to Top */}
            <Button
              variant="outline"
              onClick={scrollToTop}
              className="group flex items-center gap-2 text-sm font-semibold text-light-muted-text dark:text-dark-muted-text hover:text-primary-main transition-colors"
            >
              Back to Top
              <ArrowUp className="h-4 w-4 transform group-hover:-translate-y-1 transition-transform" />
            </Button>
          </div>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
