// src/pages/Home.jsx

import React, { useEffect, useRef, useState, useMemo } from "react";
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
import { cn } from "../lib/utils";

const projectData = {
    name: "Drishtiksha",
    tagline: "Authenticity in the Age of AI.",
    description:
        "A state-of-the-art platform leveraging a multi-model AI architecture to deliver fast, accurate, and detailed forensic analysis of digital media.",
};
const stats = [
    { icon: Star, value: "98%+", label: "Detection Accuracy" },
    { icon: Clock, value: "< 60s", label: "Average Analysis Time" },
    { icon: BrainCircuit, value: "5+", label: "Specialized AI Models" },
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
        socials: { linkedin: "https://www.linkedin.com/in/oumgadani/" },
    },
    {
        name: "Raj Mathuria",
        role: "AI/ML Developer",
        avatarUrl:
            "https://res.cloudinary.com/dcsvkcoym/image/upload/v1755573128/Member_3_yzqouc.jpg",
        socials: {
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
// Main Component
const Home = () => {
    const containerRef = useRef(null);
    const [currentSection, setCurrentSection] = useState(0);
    const [isScrolling, setIsScrolling] = useState(false);

    const sections = useMemo(
        () => ["hero", "features", "models", "tech", "team"],
        []
    );
    return (
        <div className="bg-light-background dark:bg-dark-background">
            {/* Section Navigation Dots
            <div className="fixed right-6 top-1/2 -translate-y-1/2 z-50 flex flex-col gap-3">
                {sections.map((section, index) => (
                    <button
                        key={section}
                        onClick={() => {
                            if (!isScrolling && index !== currentSection) {
                                setIsScrolling(true);
                                setCurrentSection(index);
                                setTimeout(() => setIsScrolling(false), 800);
                            }
                        }}
                        className={cn(
                            "w-3 h-3 rounded-full border-2 transition-all duration-300 hover:scale-125",
                            currentSection === index
                                ? "bg-primary-main border-primary-main"
                                : "bg-transparent border-primary-main/50 hover:border-primary-main"
                        )}
                        aria-label={`Go to ${section} section`}
                    />
                ))}
            </div> */}

            {/* Sections Container */}
            <div
                ref={containerRef}
                className="transition-transform duration-700 ease-in-out"
            >
                {/* Hero Section */}
                <section
                    id="hero"
                    className="w-full h-[90vh] flex flex-col justify-center items-center text-center px-4 sm:px-6 lg:px-8"
                >
                    <div className="w-full max-w-6xl">
                        <h1 className="text-4xl sm:text-5xl lg:text-7xl font-bold tracking-tighter mb-6 px-2">
                            {projectData.name}
                            <span className="block text-primary-main mt-2">
                                {projectData.tagline}
                            </span>
                        </h1>
                        <p className="max-w-3xl mx-auto text-lg sm:text-xl text-light-muted-text dark:text-dark-muted-text mb-12 px-4">
                            {projectData.description}
                        </p>
                        <div className="flex flex-col sm:flex-row gap-4 justify-center">
                            <Button asChild size="lg">
                                <Link to="/auth?view=login">
                                    <Play className="mr-2 h-5 w-5" /> Start
                                    Analysis
                                </Link>
                            </Button>
                            <Button asChild variant="outline" size="lg">
                                <a
                                    href="https://github.com/zaptrixio-cyber/Drishtiksha"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                >
                                    <Github className="mr-2 h-5 w-5" /> View on
                                    GitHub
                                </a>
                            </Button>
                        </div>
                    </div>
                    {/* Stats positioned at the bottom */}
                    <div className="w-full max-w-5xl mx-auto mt-16 px-4">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                            {stats.map((stat) => (
                                <div key={stat.label} className="text-center">
                                    <stat.icon className="h-8 w-8 text-primary-main mx-auto mb-3" />
                                    <div className="text-3xl font-bold">
                                        {stat.value}
                                    </div>
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
                    className="font-sansw-full h-[90vh] flex flex-col justify-center items-center px-4 sm:px-6 lg:px-8 bg-light-muted-background dark:bg-dark-muted-background"
                >
                    <div className="w-full text-center gap-4">
                        <div className="mb-4">
                            <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight mb-4">
                                Cutting-Edge Features
                            </h2>
                            <p className="text-lg sm:text-xl text-light-muted-text dark:text-dark-muted-text max-w-5xl mx-auto px-2">
                                Powered by advanced AI models and a modern
                                architecture for unparalleled detection.
                            </p>
                        </div>
                        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-10 w-full mt-12">
                            {features.map((feature) => (
                                <div key={feature.title}>
                                    <Card
                                        className={cn(
                                            "bg-light-background dark:bg-dark-background max-w-xl text-center h-full w-full transition-all duration-300",
                                            "hover:shadow-xl hover:border-primary-main/30"
                                        )}
                                    >
                                        <CardContent className="space-y-4">
                                            <div className="mx-auto w-16 h-16 bg-primary-main/10 rounded-full flex items-center justify-center">
                                                <feature.icon className="w-8 h-8 text-primary-main" />
                                            </div>
                                            <h3 className="text-xl font-semibold">
                                                {feature.title}
                                            </h3>
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
                    className="font-sans w-full min-h-screen flex flex-col justify-center items-center p-8"
                >
                    <div className="w-full text-center">
                        <div className="mb-4">
                            <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight mb-4">
                                AI Model Arsenal
                            </h2>
                            <p className="text-lg sm:text-xl text-light-muted-text dark:text-dark-muted-text max-w-5xl mx-auto px-2">
                                Multiple specialized models working together for
                                comprehensive detection.
                            </p>
                        </div>
                        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 w-full mt-12">
                            {models.map((model) => (
                                <div key={model.name}>
                                    <Card
                                        className={cn(
                                            "text-center h-full transition-all duration-300",
                                            "hover:shadow-xl hover:border-primary-main/30"
                                        )}
                                    >
                                        <CardHeader className="p-3">
                                            <div className="mx-auto w-16 h-16 bg-primary-main/10 rounded-full flex items-center justify-center">
                                                <model.icon className="w-8 h-8 text-primary-main" />
                                            </div>
                                            <h3 className="text-xl font-semibold">
                                                {model.name}
                                            </h3>
                                            <p>
                                                Model Version: v{model.version}
                                            </p>
                                        </CardHeader>
                                        <CardContent className="space-y-2">
                                            <p className="text-light-muted-text dark:text-dark-muted-text">
                                                {model.description}
                                            </p>
                                            <div className="flex items-center justify-center gap-2 text-sm">
                                                <CheckCircle className="w-4 h-4 text-green-500" />
                                                <span>
                                                    Accuracy: {model.accuracy}
                                                </span>
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
                    className="font-sansw-full h-[90vh] flex flex-col justify-center items-center px-4 sm:px-6 lg:px-8 bg-light-muted-background dark:bg-dark-muted-background"
                >
                    <div className="w-full max-w-7xl text-center">
                        <div className="mb-4">
                            <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight mb-4">
                                Our Technology Stack
                            </h2>
                            <p className="text-lg sm:text-xl text-light-muted-text dark:text-dark-muted-text max-w-3xl mx-auto px-2">
                                Built with modern, scalable technologies for
                                optimal performance and reliability.
                            </p>
                        </div>
                        <div className="relative overflow-hidden w-full mt-12 rounded-full">
                            <div className="flex animate-marquee whitespace-nowrap p-4">
                                {[...techStack, ...techStack].map(
                                    (tech, index) => (
                                        <div
                                            key={index}
                                            className="flex-shrink-0 px-12 text-center"
                                        >
                                            <tech.icon className="w-20 h-20 text-light-muted-text dark:text-dark-muted-text mx-auto hover:text-light-highlight dark:hover:text-dark-highlight dark:hover:scale-110 transition-transform duration-200 hover:scale-110" />
                                            <p className="mt-5 text-md font-bold">
                                                {tech.name}
                                            </p>
                                        </div>
                                    )
                                )}
                            </div>
                        </div>
                    </div>
                </section>

                {/* Team Section */}
                <section
                    id="team"
                    className="font-sansw-full h-[90vh] flex flex-col justify-center items-center px-4 sm:px-6 lg:px-8"
                >
                    <div className="w-full max-w-6xl text-center">
                        <div className="mb-4">
                            <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight mb-4">
                                Meet The Team
                            </h2>
                            <p className="text-lg sm:text-xl text-light-muted-text dark:text-dark-muted-text max-w-6xl mx-auto px-2">
                                Passionate experts dedicated to building the
                                future of digital media authenticity.
                            </p>
                        </div>
                        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-8 w-full mt-12">
                            {team.map((member) => (
                                <Card
                                    key={member.name}
                                    className="text-center p-6 w-full"
                                >
                                    <img
                                        src={member.avatarUrl}
                                        alt={member.name}
                                        className="w-32 h-32 rounded-full mx-auto mb-4 border-4 border-light-secondary dark:border-dark-secondary"
                                    />
                                    <h3 className="text-lg font-semibold">
                                        {member.name}
                                    </h3>
                                    <p className="text-primary-main mb-4">
                                        {member.role}
                                    </p>
                                    <div className="flex justify-center space-x-3">
                                        {member.socials.github && (
                                            <a
                                                href={member.socials.github}
                                                className="text-light-muted-text hover:text-primary-main"
                                            >
                                                <Github />
                                            </a>
                                        )}
                                        {member.socials.linkedin && (
                                            <a
                                                href={member.socials.linkedin}
                                                className="text-light-muted-text hover:text-primary-main"
                                            >
                                                <Linkedin />
                                            </a>
                                        )}
                                    </div>
                                </Card>
                            ))}
                        </div>
                    </div>
                </section>
            </div>
        </div>
    );
};

export default Home;
