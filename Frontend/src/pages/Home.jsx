// src/pages/Home.jsx

import React, { useRef, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  BrainCircuit,
  Zap,
  ShieldCheck,
  Github,
  Linkedin,
  Play,
  Star,
  Clock,
  CheckCircle,
  DatabaseZap,
  LayoutGrid,
  Blend,
} from "lucide-react";
import {
  SiReact,
  SiVite,
  SiTailwindcss,
  SiNodedotjs,
  SiExpress,
  SiPrisma,
  SiFastapi,
  SiPytorch,
  SiRedis,
  SiDocker,
} from "react-icons/si";
import { Button } from "../components/ui/Button";
import { Card, CardContent, CardHeader } from "../components/ui/Card";
import { config } from "../config/env.js";

const projectData = {
  name: config.VITE_PROJECT_NAME || "Drishtiksha",
  tagline: "Authenticity in the Age of AI.",
  description:
    "A state-of-the-art platform leveraging a multi-model AI architecture to deliver fast, accurate, and detailed forensic analysis of digital media.",
};

const stats = [
  { icon: Star, value: "80%+", label: "Detection Accuracy" },
  { icon: Clock, value: "< 60s", label: "Average Analysis Time" },
  { icon: BrainCircuit, value: "10+", label: "Specialized AI Models" },
];

const features = [
  {
    icon: BrainCircuit,
    title: "Advanced Multi-Model Analysis",
    description:
      "Our system utilizes a suite of diverse AI models to perform a multi-faceted analysis, ensuring higher accuracy and more reliable detection results.",
  },
  {
    icon: Zap,
    title: "Asynchronous Real-Time Workflow",
    description:
      "An asynchronous, queue-based architecture provides immediate feedback, while real-time updates keep you informed on analysis progress from start to finish.",
  },
  {
    icon: ShieldCheck,
    title: "Comprehensive Forensic Reports",
    description:
      "Go beyond a simple verdict. Dive deep with frame-by-frame confidence charts, processing metadata, and downloadable PDF reports for thorough documentation.",
  },
  {
    icon: DatabaseZap,
    title: "Model-Agnostic Backend",
    description:
      "Our decoupled backend automatically detects and integrates with all available ML models, ensuring seamless scalability and future-proofing the platform.",
  },
  {
    icon: LayoutGrid,
    title: "Intuitive & Responsive Interface",
    description:
      "A clean, modern user interface built with a responsive design ensures a seamless experience across all devices, from desktops to mobile phones.",
  },
  {
    icon: Blend,
    title: "Customizable Dark & Light Themes",
    description:
      "Choose your preferred viewing experience. The system intelligently adapts to your OS settings or allows you to toggle between themes manually.",
  },
];

const models = [
  {
    name: "SigLip-LSTM",
    version: "4",
    description:
      "The latest and most robust model, featuring deeper classifier heads and dropout for state-of-the-art accuracy and generalization.",
    specialty: "State-of-the-Art & High Accuracy",
    icon: BrainCircuit,
    accuracy: "98%",
  },
  {
    name: "EfficientNet-B7",
    version: "1",
    description:
      "A powerful classifier that inspects every frame for spatial artifacts by analyzing each detected face individually.",
    specialty: "Spatial Artifacts & Per-Face Analysis",
    icon: Zap,
    accuracy: "98%",
  },
  {
    name: "Color Cues LSTM",
    version: "1",
    description:
      "A specialized model engineered to detect subtle color inconsistencies and artifacts often left behind by deepfake generation algorithms.",
    specialty: "Color & Artifact Detection",
    accuracy: "75%",
    icon: LayoutGrid,
  },
  {
    name: "Eyeblink CNN+LSTM",
    version: "1 - Inference Only",
    description:
      "A specialized model that focuses on the biological cue of eye blinking patterns to detect inconsistencies often found in deepfakes.",
    specialty: "Blink Pattern & Behavioral Analysis",
    icon: CheckCircle,
    accuracy: "90%",
  },
  {
    name: "SigLip-LSTM",
    version: "3 (Legacy)",
    description:
      "An advanced model combining a powerful vision encoder with temporal analysis for high-stakes forensic investigation.",
    specialty: "High Accuracy & Temporal Analysis",
    accuracy: "60%",
    icon: BrainCircuit,
  },
  {
    name: "SigLip-LSTM",
    version: "1 (Legacy) - Inference Only",
    description:
      "The foundational model offering a strong balance of speed and accuracy, perfect for general-purpose and real-time screening.",
    specialty: "Balanced Performance",
    accuracy: "60%",
    icon: ShieldCheck,
  },
];

const team = [
  {
    name: "Kandarp Gajjar",
    role: "AI/ML & Full-Stack Developer",
    avatarUrl:
      "https://res.cloudinary.com/dcsvkcoym/image/upload/v1755573128/Member_1_erlley.jpg",
    socials: {
      github: "https://github.com/slantie",
      linkedin: "https://www.linkedin.com/in/kandarpgajjar/",
    },
  },
  {
    name: "Oum Gadani",
    role: "AI/ML Developer",
    avatarUrl:
      "https://res.cloudinary.com/dcsvkcoym/image/upload/v1755573128/Member_2_mnpfhh.jpg",
    socials: {
      github: "https://github.com/Oum-Gadani",
      linkedin: "https://www.linkedin.com/in/oumgadani/",
    },
  },
  {
    name: "Raj Mathuria",
    role: "AI/ML Developer",
    avatarUrl:
      "https://res.cloudinary.com/dcsvkcoym/image/upload/v1755573128/Member_3_yzqouc.jpg",
    socials: {
      github: "https://github.com/CodeCraftsmanRaj",
      linkedin: "https://www.linkedin.com/in/raj-mathuria-98a710283/",
    },
  },
  {
    name: "Vishwajit Sarnobat",
    role: "AI/ML Developer",
    avatarUrl:
      "https://res.cloudinary.com/dcsvkcoym/image/upload/v1755573129/Member_4_tntyts.jpg",
    socials: {
      github: "https://github.com/vishwajitsarnobat",
      linkedin: "https://www.linkedin.com/in/vishwajitsarnobat/",
    },
  },
];

const techStack = [
  { icon: SiReact, name: "React" },
  { icon: SiVite, name: "Vite" },
  { icon: SiTailwindcss, name: "Tailwind CSS" },
  { icon: SiNodedotjs, name: "Node.js" },
  { icon: SiExpress, name: "Express" },
  { icon: SiPrisma, name: "Prisma" },
  { icon: SiFastapi, name: "FastAPI" },
  { icon: SiPytorch, name: "PyTorch" },
  { icon: SiRedis, name: "Redis" },
  { icon: SiDocker, name: "Docker" },
];

const Home = () => {
  const [visibleSections, setVisibleSections] = useState(new Set());
  const sectionsRef = useRef({});

  useEffect(() => {
    document.title = "Drishtiksha - AI-Powered Media Analysis";

    // Intersection Observer for scroll-triggered animations
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setVisibleSections((prev) => new Set([...prev, entry.target.id]));
          }
        });
      },
      { threshold: 0.1 }
    );

    Object.values(sectionsRef.current).forEach((section) => {
      if (section) observer.observe(section);
    });

    return () => observer.disconnect();
  }, []);

  const isVisible = (id) => visibleSections.has(id);

  return (
    <div className="bg-light-background dark:bg-dark-background text-light-text dark:text-dark-text min-h-screen">
      {/* Hero Section */}
      <section
        id="hero"
        ref={(el) => (sectionsRef.current.hero = el)}
        className="w-full min-h-screen flex flex-col justify-center items-center text-center py-24 px-4 sm:px-6 lg:px-8 space-y-12"
      >
        <div className="max-w-screen-lg mx-auto animate-in fade-in slide-in-from-bottom-8 duration-700">
          <h1 className="text-4xl sm:text-5xl lg:text-7xl font-bold tracking-tighter mb-6">
            {projectData.name}
            <span className="block text-primary-main mt-4 animate-in fade-in slide-in-from-bottom-4 duration-700 delay-150">
              {projectData.tagline}
            </span>
          </h1>
          <p className="text-lg sm:text-xl text-light-muted-text dark:text-dark-muted-text max-w-3xl mx-auto mb-12 animate-in fade-in slide-in-from-bottom-4 duration-700 delay-300">
            {projectData.description}
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center animate-in fade-in slide-in-from-bottom-4 duration-700 delay-500">
            <Button
              asChild
              size="lg"
              className="transition-all duration-200 hover:scale-105 active:scale-95"
            >
              <Link to="/auth?view=login">
                <Play className="mr-2 h-5 w-5" /> Start Analysis
              </Link>
            </Button>
          </div>
        </div>

        {/* Stats */}
        <div className="w-full max-w-screen-xl mx-auto mt-16 px-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 md:gap-12">
            {stats.map((stat, index) => (
              <div
                key={stat.label}
                className="text-center animate-in fade-in slide-in-from-bottom-4 duration-700"
                style={{ animationDelay: `${600 + index * 150}ms` }}
              >
                <div className="inline-block p-4 bg-primary-main/10 rounded-full mb-3 transition-transform duration-300 hover:scale-110">
                  <stat.icon className="h-10 w-10 text-primary-main" />
                </div>
                <div className="text-3xl font-bold">{stat.value}</div>
                <div className="text-sm text-light-muted-text dark:text-dark-muted-text">
                  {stat.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section
        id="features"
        ref={(el) => (sectionsRef.current.features = el)}
        className="w-full min-h-screen flex flex-col justify-center items-center py-24 px-4 sm:px-6 lg:px-8 bg-light-muted-background dark:bg-dark-muted-background space-y-12"
      >
        <div className="max-w-screen-xl mx-auto text-center">
          <div
            className={`mb-4 transition-all duration-700 ${
              isVisible("features")
                ? "opacity-100 translate-y-0"
                : "opacity-0 translate-y-8"
            }`}
          >
            <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight mb-4 text-primary-main">
              Cutting-Edge Features
            </h2>
            <p className="text-lg sm:text-xl text-light-muted-text dark:text-dark-muted-text max-w-3xl mx-auto">
              Powered by advanced AI models and a modern architecture for
              unparalleled detection.
            </p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mt-12">
            {features.map((feature, index) => (
              <div
                key={feature.title}
                className={`transition-all duration-700 ${
                  isVisible("features")
                    ? "opacity-100 translate-y-0"
                    : "opacity-0 translate-y-8"
                }`}
                style={{ transitionDelay: `${index * 100}ms` }}
              >
                <Card className="bg-light-background dark:bg-dark-background text-center h-full w-full transition-all duration-300 hover:shadow-xl hover:border-primary-main/30 hover:-translate-y-2">
                  <CardContent className="space-y-4 py-8">
                    <div className="mx-auto w-16 h-16 bg-primary-main/10 rounded-full flex items-center justify-center transition-transform duration-300 hover:scale-110">
                      <feature.icon className="w-8 h-8 text-primary-main" />
                    </div>
                    <h3 className="text-xl font-semibold">{feature.title}</h3>
                    <p className="text-light-muted-text dark:text-dark-muted-text">
                      {feature.description}
                    </p>
                  </CardContent>
                </Card>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Models Section */}
      <section
        id="models"
        ref={(el) => (sectionsRef.current.models = el)}
        className="w-full min-h-screen flex flex-col justify-center items-center py-24 px-4 sm:px-6 lg:px-8 space-y-12"
      >
        <div className="max-w-screen-xl mx-auto text-center">
          <div
            className={`mb-4 transition-all duration-700 ${
              isVisible("models")
                ? "opacity-100 translate-y-0"
                : "opacity-0 translate-y-8"
            }`}
          >
            <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight mb-4 text-primary-main">
              AI Model Arsenal
            </h2>
            <p className="text-lg sm:text-xl text-light-muted-text dark:text-dark-muted-text max-w-3xl mx-auto">
              Multiple specialized models working together for comprehensive
              detection.
            </p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mt-12">
            {models.map((model, index) => (
              <div
                key={model.name + model.version}
                className={`transition-all duration-700 ${
                  isVisible("models")
                    ? "opacity-100 translate-y-0"
                    : "opacity-0 translate-y-8"
                }`}
                style={{ transitionDelay: `${index * 100}ms` }}
              >
                <Card className="text-center h-full transition-all duration-300 hover:shadow-xl hover:border-primary-main/30 hover:-translate-y-2 group">
                  <CardHeader className="p-6">
                    <div className="mx-auto w-16 h-16 bg-primary-main/10 rounded-full flex items-center justify-center transition-transform duration-300 group-hover:scale-110">
                      <model.icon className="w-8 h-8 text-primary-main" />
                    </div>
                    <h3 className="text-xl font-semibold">{model.name}</h3>
                    <p className="text-light-muted-text dark:text-dark-muted-text text-sm">
                      Model Version: v{model.version}
                    </p>
                  </CardHeader>
                  <CardContent className="space-y-3 py-6">
                    <p className="text-light-muted-text dark:text-dark-muted-text text-base">
                      {model.description}
                    </p>
                    <div className="flex items-center justify-center gap-2 text-sm text-green-500 font-semibold">
                      <CheckCircle className="w-4 h-4" />
                      <span>Accuracy: {model.accuracy}</span>
                    </div>
                  </CardContent>
                </Card>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Tech Stack Section */}
      <section
        id="tech"
        ref={(el) => (sectionsRef.current.tech = el)}
        className="w-full min-h-screen flex flex-col justify-center items-center py-24 px-4 sm:px-6 lg:px-8 bg-light-muted-background dark:bg-dark-muted-background space-y-12"
      >
        <div className="w-full max-w-screen-xl mx-auto text-center">
          <div
            className={`mb-4 transition-all duration-700 ${
              isVisible("tech")
                ? "opacity-100 translate-y-0"
                : "opacity-0 translate-y-8"
            }`}
          >
            <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight mb-4 text-primary-main">
              Our Technology Stack
            </h2>
            <p className="text-lg sm:text-xl text-light-muted-text dark:text-dark-muted-text max-w-3xl mx-auto">
              Built with modern, scalable technologies for optimal performance
              and reliability.
            </p>
          </div>
          {/* Marquee Animation - Desktop */}
          <div className="hidden md:block relative overflow-hidden w-full mt-12 py-8 rounded-full border border-light-secondary dark:border-dark-secondary bg-light-background dark:bg-dark-background">
            <div className="flex animate-marquee whitespace-nowrap">
              {[...techStack, ...techStack].map((tech, index) => (
                <div
                  key={index}
                  className="flex-shrink-0 flex flex-col items-center justify-center px-10 text-center group"
                >
                  <tech.icon className="w-20 h-20 text-light-muted-text dark:text-dark-muted-text mx-auto transition-all duration-300 group-hover:text-primary-main group-hover:-translate-y-1" />
                  <p className="mt-5 text-md font-bold group-hover:text-primary-main">
                    {tech.name}
                  </p>
                </div>
              ))}
            </div>
          </div>
          {/* Grid Layout - Mobile */}
          <div className="md:hidden grid grid-cols-2 sm:grid-cols-3 gap-8 mt-12">
            {techStack.map((tech, index) => (
              <div
                key={tech.name}
                className={`flex flex-col items-center justify-center text-center transition-all duration-700 ${
                  isVisible("tech")
                    ? "opacity-100 translate-y-0"
                    : "opacity-0 translate-y-8"
                }`}
                style={{ transitionDelay: `${index * 50}ms` }}
              >
                <div className="p-4 bg-light-background dark:bg-dark-background rounded-xl border border-light-secondary dark:border-dark-secondary transition-all duration-300 hover:border-primary-main/50 hover:shadow-lg">
                  <tech.icon className="w-12 h-12 sm:w-16 sm:h-16 text-light-muted-text dark:text-dark-muted-text transition-all duration-300 hover:text-primary-main" />
                </div>
                <p className="mt-3 text-sm font-bold">{tech.name}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section
        id="team"
        ref={(el) => (sectionsRef.current.team = el)}
        className="w-full min-h-screen flex flex-col justify-center items-center py-24 px-4 sm:px-6 lg:px-8 space-y-12"
      >
        <div className="max-w-screen-xl mx-auto text-center">
          <div
            className={`mb-4 transition-all duration-700 ${
              isVisible("team")
                ? "opacity-100 translate-y-0"
                : "opacity-0 translate-y-8"
            }`}
          >
            <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight mb-4 text-primary-main">
              Meet The Team
            </h2>
            <p className="text-lg sm:text-xl text-light-muted-text dark:text-dark-muted-text max-w-3xl mx-auto">
              Passionate experts dedicated to building the future of digital
              media authenticity.
            </p>
          </div>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-8 mt-12">
            {team.map((member, index) => (
              <div
                key={member.name}
                className={`transition-all duration-700 ${
                  isVisible("team")
                    ? "opacity-100 translate-y-0"
                    : "opacity-0 translate-y-8"
                }`}
                style={{ transitionDelay: `${index * 100}ms` }}
              >
                <Card className="text-center p-6 w-full h-full flex flex-col items-center justify-center transition-all duration-300 hover:shadow-xl hover:border-primary-main/30 hover:-translate-y-2 group">
                  <div className="relative mb-4">
                    <img
                      src={member.avatarUrl}
                      alt={member.name}
                      className="w-32 h-32 rounded-full mx-auto border-4 border-primary-main/20 object-cover transition-all duration-300 group-hover:border-primary-main/50 group-hover:scale-105"
                    />
                    <div className="absolute inset-0 rounded-full bg-primary-main/0 group-hover:bg-primary-main/10 transition-all duration-300" />
                  </div>
                  <h3 className="text-lg font-semibold">{member.name}</h3>
                  <p className="text-primary-main mb-4">{member.role}</p>
                  <div className="flex justify-center space-x-4">
                    {member.socials.github && (
                      <a
                        href={member.socials.github}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-light-muted-text hover:text-primary-main transition-all duration-200 hover:scale-125"
                        aria-label={`${member.name} on GitHub`}
                      >
                        <Github className="h-6 w-6" />
                      </a>
                    )}
                    {member.socials.linkedin && (
                      <a
                        href={member.socials.linkedin}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-light-muted-text hover:text-primary-main transition-all duration-200 hover:scale-125"
                        aria-label={`${member.name} on LinkedIn`}
                      >
                        <Linkedin className="h-6 w-6" />
                      </a>
                    )}
                  </div>
                </Card>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;
