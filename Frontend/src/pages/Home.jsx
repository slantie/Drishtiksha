import React from "react";
import {
  ArrowRight,
  Code,
  Palette,
  Zap,
  Settings,
  Star,
  GitBranch,
  Users,
  Activity,
} from "lucide-react";

const projectName = import.meta.env.VITE_PROJECT_NAME;

function Home() {
  return (
    <div className="">
      <div className="flex items-center justify-between space-x-48">
        <div className="text-left flex-1">
          <div className="">
            <h1 className="text-5xl md:text-6xl font-bold mb-6 text-light-text dark:text-dark-text leading-tight">
              Welcome to{" "}
              <span className="bg-gradient-to-r from-light-highlight to-light-highlight/80 dark:from-dark-highlight dark:to-dark-highlight/80 bg-clip-text text-transparent">
                {projectName}
              </span>
            </h1>

            <p className="text-xl text-light-muted-text dark:text-dark-muted-text mb-8 leading-relaxed">
              Transform your workflow with our powerful, intuitive platform
              designed for modern teams and individuals.
              <span className="text-light-text dark:text-dark-text font-semibold">
                {" "}
                Built for speed, designed for scale.
              </span>
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-start items-start">
              <a
                href="/login"
                className="group bg-light-highlight dark:bg-dark-highlight text-white font-semibold py-4 px-10 rounded-xl hover:shadow-lg hover:shadow-light-highlight/25 dark:hover:shadow-dark-highlight/25 transition-all duration-300 flex items-center space-x-2"
              >
                <span>Get Started</span>
                <ArrowRight className="w-5 h-5 group-hover:translate-x-2 transition-transform" />
              </a>
              <a
                href="/setup"
                className="bg-light-muted-background dark:bg-dark-muted-background border-2 border-light-muted-text/20 dark:border-dark-muted-text/20 hover:border-light-highlight dark:hover:border-dark-highlight font-semibold py-4 px-10 rounded-xl transition-all duration-300 hover:shadow-md"
              >
                View Documentation
              </a>
            </div>
          </div>
        </div>

        <div className="hidden lg:block flex-shrink-0">
          <div className="relative">
            <div className="w-[44rem] bg-gradient-to-br from-light-muted-background to-light-background dark:from-dark-muted-background dark:to-dark-background rounded-3xl p-8 shadow-2xl border border-light-muted-text/10 dark:border-dark-muted-text/10 backdrop-blur-sm">
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 rounded-full bg-red-400"></div>
                    <div className="w-3 h-3 rounded-full bg-yellow-400"></div>
                    <div className="w-3 h-3 rounded-full bg-green-400"></div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="flex items-center space-x-1 bg-light-background dark:bg-dark-background px-3 py-1 rounded-full">
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                      <span className="text-xs text-light-muted-text dark:text-dark-muted-text">
                        Live
                      </span>
                    </div>
                    <Users className="w-4 h-4 text-light-muted-text dark:text-dark-muted-text" />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-light-background dark:bg-dark-background p-5 rounded-xl border border-light-muted-text/10 dark:border-dark-muted-text/10 hover:border-light-highlight/30 dark:hover:border-dark-highlight/30 transition-all duration-300 group hover:shadow-lg hover:-translate-y-1">
                    <div className="flex items-center justify-between mb-3">
                      <Code className="w-7 h-7 text-light-highlight dark:text-dark-highlight group-hover:scale-110 transition-transform" />
                      <div className="text-xs text-light-muted-text dark:text-dark-muted-text bg-light-muted-background dark:bg-dark-muted-background px-2 py-1 rounded">
                        99.9%
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="h-2 bg-light-noisy-background dark:bg-dark-noisy-background rounded"></div>
                      <div className="h-2 bg-light-noisy-background dark:bg-dark-noisy-background rounded w-3/4"></div>
                      <div className="flex justify-between items-center pt-1">
                        <span className="text-xs text-light-muted-text dark:text-dark-muted-text">
                          Code Quality
                        </span>
                        <Activity className="w-3 h-3 text-green-400" />
                      </div>
                    </div>
                  </div>

                  <div className="bg-light-background dark:bg-dark-background p-5 rounded-xl border border-light-muted-text/10 dark:border-dark-muted-text/10 hover:border-light-highlight/30 dark:hover:border-dark-highlight/30 transition-all duration-300 group hover:shadow-lg hover:-translate-y-1">
                    <div className="flex items-center justify-between mb-3">
                      <Palette className="w-7 h-7 text-light-highlight dark:text-dark-highlight group-hover:scale-110 transition-transform" />
                      <GitBranch className="w-4 h-4 text-light-muted-text dark:text-dark-muted-text" />
                    </div>
                    <div className="space-y-2">
                      <div className="h-2 bg-light-noisy-background dark:bg-dark-noisy-background rounded"></div>
                      <div className="h-2 bg-light-noisy-background dark:bg-dark-noisy-background rounded w-2/3"></div>
                      <div className="flex justify-between items-center pt-1">
                        <span className="text-xs text-light-muted-text dark:text-dark-muted-text">
                          Design System
                        </span>
                        <div className="text-xs text-green-400">Active</div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-light-background dark:bg-dark-background p-5 rounded-xl border border-light-muted-text/10 dark:border-dark-muted-text/10 hover:border-light-highlight/30 dark:hover:border-dark-highlight/30 transition-all duration-300 group hover:shadow-lg hover:-translate-y-1">
                    <div className="flex items-center justify-between mb-3">
                      <Zap className="w-7 h-7 text-light-highlight dark:text-dark-highlight group-hover:scale-110 transition-transform" />
                      <div className="text-xs text-light-muted-text dark:text-dark-muted-text">
                        2.3s
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="h-2 bg-light-highlight/20 dark:bg-dark-highlight/20 rounded overflow-hidden">
                        <div
                          className="h-full bg-light-highlight dark:bg-dark-highlight rounded animate-pulse"
                          style={{ width: "85%" }}
                        ></div>
                      </div>
                      <div className="h-2 bg-light-noisy-background dark:bg-dark-noisy-background rounded w-4/5"></div>
                      <div className="flex justify-between items-center pt-1">
                        <span className="text-xs text-light-muted-text dark:text-dark-muted-text">
                          Performance
                        </span>
                        <div className="text-xs text-green-400">Optimal</div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-light-background dark:bg-dark-background p-5 rounded-xl border border-light-muted-text/10 dark:border-dark-muted-text/10 hover:border-light-highlight/30 dark:hover:border-dark-highlight/30 transition-all duration-300 group hover:shadow-lg hover:-translate-y-1">
                    <div className="flex items-center justify-between mb-3">
                      <Settings className="w-7 h-7 text-light-highlight dark:text-dark-highlight group-hover:scale-110 transition-transform" />
                      <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                    </div>
                    <div className="space-y-2">
                      <div className="h-2 bg-light-noisy-background dark:bg-dark-noisy-background rounded"></div>
                      <div className="h-2 bg-light-noisy-background dark:bg-dark-noisy-background rounded w-3/5"></div>
                      <div className="flex justify-between items-center pt-1">
                        <span className="text-xs text-light-muted-text dark:text-dark-muted-text">
                          Configuration
                        </span>
                        <div className="text-xs text-blue-400">Syncing</div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-gradient-to-r from-light-background to-light-muted-background dark:from-dark-background dark:to-dark-muted-background p-5 rounded-xl border border-light-muted-text/10 dark:border-dark-muted-text/10">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-gradient-to-r from-light-highlight to-light-highlight/80 dark:from-dark-highlight dark:to-dark-highlight/80 rounded-xl flex items-center justify-center">
                        <Activity className="w-5 h-5 text-white" />
                      </div>
                      <div>
                        <div className="text-sm font-semibold text-light-text dark:text-dark-text">
                          System Health
                        </div>
                        <div className="text-xs text-light-muted-text dark:text-dark-muted-text">
                          All services running
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-semibold text-green-400">
                        98.7%
                      </div>
                      <div className="text-xs text-light-muted-text dark:text-dark-muted-text">
                        Uptime
                      </div>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-light-muted-text dark:text-dark-muted-text">
                        CPU Usage
                      </span>
                      <span className="text-light-text dark:text-dark-text">
                        23%
                      </span>
                    </div>
                    <div className="h-2 bg-light-noisy-background dark:bg-dark-noisy-background rounded overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-light-highlight to-green-400 dark:from-dark-highlight dark:to-green-400 rounded animate-pulse"
                        style={{ width: "23%" }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="absolute -top-6 -right-6 w-12 h-12 bg-gradient-to-r from-light-highlight to-light-highlight/80 dark:from-dark-highlight dark:to-dark-highlight/80 rounded-2xl shadow-lg flex items-center justify-center cursor-pointer group hover:scale-110 transition-all duration-300 animate-bounce hover:animate-spin">
              <Star className="w-6 h-6 text-white group-hover:rotate-180 transition-transform duration-500" />
            </div>
            <div className="absolute -bottom-4 -left-4 w-8 h-8 bg-gradient-to-r from-blue-400 to-purple-400 rounded-xl shadow-lg cursor-pointer hover:rotate-45 hover:scale-125 transition-all duration-500 animate-pulse hover:animate-bounce"></div>
            <div className="absolute top-1/3 -left-8 w-6 h-6 bg-gradient-to-r from-green-400 to-blue-400 rounded-full cursor-pointer hover:scale-150 hover:rounded-xl transition-all duration-300 animate-ping hover:animate-pulse"></div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Home;
