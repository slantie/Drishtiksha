// src/pages/Home.jsx

import React from "react";
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

// --- Static Data (Preserved) ---
const projectData = {
    name: "Drishtiksha",
    tagline: "Authenticity in the Age of AI.",
    description:
        "A state-of-the-art platform leveraging a multi-model AI architecture to deliver fast, accurate, and detailed forensic analysis of digital media.",
};
const stats = [
    { icon: Star, value: "80%+", label: "Detection Accuracy" },
    { icon: Clock, value: "< 30s", label: "Average Analysis Time" },
    { icon: BrainCircuit, value: "2", label: "Specialized AI Models" },
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
];
const models = [
    {
        name: "SIGLIP-LSTM V3",
        version: "3.0.0",
        description:
            "Our most advanced model, combining a powerful vision encoder with temporal analysis for high-stakes forensic investigation.",
        specialty: "High Accuracy & Temporal Analysis",
    },
    {
        name: "Color Cues LSTM V1",
        version: "1.0.0",
        description:
            "A specialized model engineered to detect subtle color inconsistencies and artifacts often left behind by deepfake generation algorithms.",
        specialty: "Color & Artifact Detection",
    },
    {
        name: "SIGLIP-LSTM V1 (Legacy)",
        version: "1.0.0",
        description:
            "The foundational model offering a strong balance of speed and accuracy, perfect for general-purpose and real-time screening.",
        specialty: "Balanced Performance",
    },
];
const team = [
    {
        name: "Kandarp Gajjar",
        role: "AI/ML & Full-Stack",
        avatarUrl: "https://avatars.githubusercontent.com/u/128211660?v=4",
        socials: {
            github: "https://github.com/slantie",
            linkedin: "https://www.linkedin.com/in/kandarpgajjar/",
        },
    },
    {
        name: "Oum Gadani",
        role: "AI/ML Developer",
        avatarUrl:
            "https://avatars.githubusercontent.com/u/128615348?v=4",
        socials: { linkedin: "https://www.linkedin.com/in/oumgadani/" },
    },
    {
        name: "Raj Mathuria",
        role: "AI/ML Developer",
        avatarUrl:
            "https://avatars.githubusercontent.com/u/128615348?v=4",
        socials: {
            linkedin: "https://www.linkedin.com/in/raj-mathuria-98a710283/",
        },
    },
    {
        name: "Vishwajit Sarnobat",
        role: "AI/ML Developer",
        avatarUrl:
            "https://avatars.githubusercontent.com/u/128615348?v=4",
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

// --- Reusable Section Component ---

// REFACTOR: The Section component now provides consistent padding and max-width, NOT a fixed height.
const Section = ({ id, title, subtitle, children, className }) => (
    <section id={id} className={cn("py-20 sm:py-24", className)}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            {(title || subtitle) && (
                <div className="max-w-6xl mx-auto text-center mb-16">
                    <h2 className="text-4xl font-bold tracking-tight">
                        {title}
                    </h2>
                    <p className="mt-4 text-lg text-light-muted-text dark:text-dark-muted-text">
                        {subtitle}
                    </p>
                </div>
            )}
            {children}
        </div>
    </section>
);

// --- Main Home Component ---

const Home = () => {
    return (
        <div className="bg-light-background dark:bg-dark-background">
            {/* REFACTOR: The Hero Section is now a unique element with min-h-screen for a full-page introduction. */}
            <section className="relative min-h-[90vh] flex flex-col justify-center items-center text-center p-4">
                <div className="w-full max-w-6xl">
                    <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold tracking-tighter mb-6">
                        {projectData.name}
                        <span className="block text-primary-main mt-2">
                            {projectData.tagline}
                        </span>
                    </h1>
                    <p className="max-w-3xl mx-auto text-xl text-light-muted-text dark:text-dark-muted-text mb-12">
                        {projectData.description}
                    </p>
                    <div className="flex flex-col sm:flex-row gap-4 justify-center">
                        <Button asChild size="lg">
                            <Link to="/auth?view=login">
                                <Play className="mr-2 h-5 w-5" /> Start Analysis
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
                {/* Stats are positioned at the bottom of the hero section */}
                <div className="w-full max-w-5xl mx-auto mt-24 pb-8">
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

            {/* REFACTOR: Subsequent sections use the reusable Section component with alternating backgrounds. */}
            <Section
                id="features"
                title="Cutting-Edge Features"
                subtitle="Powered by advanced AI models and a modern architecture for unparalleled detection."
                className="bg-light-muted-background dark:bg-dark-muted-background"
            >
                <div className="grid md:grid-cols-3 gap-8">
                    {features.map((feature) => (
                        <Card key={feature.title} className="text-center">
                            <CardHeader>
                                <div className="mx-auto w-16 h-16 bg-primary-main/10 rounded-xl flex items-center justify-center">
                                    <feature.icon className="w-8 h-8 text-primary-main" />
                                </div>
                            </CardHeader>
                            <CardContent className="space-y-2">
                                <h3 className="text-xl font-semibold">
                                    {feature.title}
                                </h3>
                                <p className="text-light-muted-text dark:text-dark-muted-text">
                                    {feature.description}
                                </p>
                            </CardContent>
                        </Card>
                    ))}
                </div>
            </Section>

            <Section
                id="models"
                title="AI Model Arsenal"
                subtitle="Multiple specialized models working together for comprehensive detection."
            >
                <div className="space-y-6">
                    {models.map((model) => (
                        <Card key={model.name} className="p-8">
                            <div className="flex flex-col lg:flex-row lg:items-center justify-between">
                                <div className="flex-1">
                                    <div className="flex items-center mb-4">
                                        <div className="inline-flex items-center justify-center w-12 h-12 bg-primary-main/10 rounded-full mr-4">
                                            <BrainCircuit className="w-6 h-6 text-primary-main" />
                                        </div>
                                        <h3 className="text-2xl font-bold mr-4">
                                            {model.name}
                                        </h3>
                                        <span className="bg-primary-main/10 text-primary-main px-3 py-1 rounded-full text-sm font-medium">
                                            v{model.version}
                                        </span>
                                    </div>
                                    <p className="text-light-muted-text dark:text-dark-text mb-4 lg:mb-0 leading-relaxed">
                                        {model.description}
                                    </p>
                                </div>
                                <div className="lg:ml-8">
                                    <div className="inline-flex items-center bg-light-hover dark:bg-dark-hover px-4 py-2 rounded-lg font-medium">
                                        <CheckCircle className="w-4 h-4 mr-2 text-green-500" />
                                        {model.specialty}
                                    </div>
                                </div>
                            </div>
                        </Card>
                    ))}
                </div>
            </Section>

            <Section
                id="tech"
                title="Our Technology Stack"
                subtitle="Built with modern, scalable technologies for optimal performance and reliability."
                className="bg-light-muted-background dark:bg-dark-muted-background"
            >
                <div className="relative overflow-hidden">
                    <div className="flex animate-marquee whitespace-nowrap">
                        {[...techStack, ...techStack].map((tech, index) => (
                            <div key={index} className="px-10 text-center">
                                <tech.icon className="w-20 h-20 p-1  text-light-muted-text dark:text-dark-muted-text mx-auto hover:text-light-highlight dark:hover:text-dark-highlight dark:hover:scale-110 transition-transform duration-200 hover:scale-110" />
                                <p className="mt-5 text-md font-bold">
                                    {tech.name}
                                </p>
                            </div>
                        ))}
                    </div>
                </div>
            </Section>

            <Section
                id="team"
                title="Meet The Team"
                subtitle="Passionate experts dedicated to building the future of digital media authenticity."
                className="min-h-[80vh] flex items-center justify-center w-screen"
            >
                <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-8 w-full">
                    {team.map((member) => (
                        <Card key={member.name} className="text-center p-6 w-full">
                            <img
                                src={member.avatarUrl}
                                alt={member.name}
                                className="w-60 h-60 rounded-full mx-auto mb-4 border-4 border-light-secondary dark:border-dark-secondary"
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
            </Section>
        </div>
    );
};

export default Home;
