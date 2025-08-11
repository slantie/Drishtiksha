import { Github, Linkedin, Heart, ArrowUp } from "lucide-react";

function Footer() {
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <footer className="bg-light-muted-background/30 dark:bg-dark-muted-background/30 border-t border-light-muted-text/10 dark:border-dark-muted-text/10">
      <div className="mx-auto px-6 py-8">
        <div className="flex flex-col md:flex-row justify-between items-center space-y-6 md:space-y-0">
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-4">
              <a
                href="https://github.com/slantie"
                target="_blank"
                rel="noopener noreferrer"
                className="group flex items-center space-x-2 text-light-muted-text dark:text-dark-muted-text hover:text-light-highlight dark:hover:text-dark-highlight transition-all duration-300 hover:scale-105"
              >
                <div className="p-2 rounded-lg bg-light-background dark:bg-dark-background border border-light-muted-text/10 dark:border-dark-muted-text/10 group-hover:border-light-highlight/30 dark:group-hover:border-dark-highlight/30 transition-all duration-300 group-hover:shadow-lg">
                  <Github className="w-4 h-4 group-hover:rotate-12 transition-transform duration-300" />
                </div>
                <span className="font-medium">GitHub</span>
              </a>
              
              <a
                href="https://linkedin.com/in/kandarpgajjar"
                target="_blank"
                rel="noopener noreferrer"
                className="group flex items-center space-x-2 text-light-muted-text dark:text-dark-muted-text hover:text-light-highlight dark:hover:text-dark-highlight transition-all duration-300 hover:scale-105"
              >
                <div className="p-2 rounded-lg bg-light-background dark:bg-dark-background border border-light-muted-text/10 dark:border-dark-muted-text/10 group-hover:border-light-highlight/30 dark:group-hover:border-dark-highlight/30 transition-all duration-300 group-hover:shadow-lg">
                  <Linkedin className="w-4 h-4 group-hover:rotate-12 transition-transform duration-300" />
                </div>
                <span className="font-medium">LinkedIn</span>
              </a>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <span className="text-light-muted-text dark:text-dark-muted-text">Made with</span>
            <Heart className="w-4 h-4 text-red-400 animate-pulse" />
            <span className="text-light-muted-text dark:text-dark-muted-text">by</span>
            <span className="font-bold text-light-highlight dark:text-dark-highlight">
              Slantie
            </span>
          </div>

          <div className="flex items-center space-x-4">
            <button
              onClick={scrollToTop}
              className="group flex items-center space-x-2 text-light-muted-text dark:text-dark-muted-text hover:text-light-highlight dark:hover:text-dark-highlight transition-all duration-300 hover:scale-105"
            >
              <div className="p-2 rounded-lg bg-light-background dark:bg-dark-background border border-light-muted-text/10 dark:border-dark-muted-text/10 group-hover:border-light-highlight/30 dark:group-hover:border-dark-highlight/30 transition-all duration-300 group-hover:shadow-lg group-hover:-translate-y-1">
                <ArrowUp className="w-4 h-4 group-hover:-translate-y-1 transition-transform duration-300" />
              </div>
              <span className="font-medium hidden sm:block">Back to Top</span>
            </button>
          </div>
        </div>
      </div>
    </footer>
  );
}

export default Footer;