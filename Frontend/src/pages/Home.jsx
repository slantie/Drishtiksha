import React, { useState, useEffect } from "react";
import {
    BrainCircuit,
    Zap,
    ShieldCheck,
    Github,
    Linkedin,
    Twitter,
    ChevronRight,
    Play,
    Star,
    Users,
    Clock,
    CheckCircle,
    ArrowUp,
    ChartCandlestickIcon,
    ScanFaceIcon,
} from "lucide-react";

import {
    SiReact,
    SiVite,
    SiTailwindcss,
    SiReactquery,
    SiNodedotjs,
    SiExpress,
    SiPrisma,
    SiPostgresql,
    SiSocketdotio,
    SiFastapi,
    SiPytorch,
    SiOpencv,
    SiRedis,
    SiDocker,
    SiCloudinary,
    SiGithubactions,
} from "react-icons/si";

const Home = () => {
    const projectData = {
        projectDetails: {
            name: "Drishtiksha",
            tagline: "Authenticity in the Age of AI.",
            description:
                "A state-of-the-art platform leveraging a multi-model AI architecture to deliver fast, accurate, and detailed forensic analysis of digital media. We provide the tools to verify authenticity and combat the spread of sophisticated deepfakes.",
        },
        features: [
            {
                icon: BrainCircuit,
                title: "Advanced Multi-Model Analysis",
                description:
                    "Our system utilizes a suite of diverse AI models, including SigLIP-LSTM and Color Cues, to perform a multi-faceted analysis, ensuring higher accuracy and more reliable detection results.",
            },
            {
                icon: Zap,
                title: "Asynchronous Real-Time Workflow",
                description:
                    "An asynchronous, queue-based architecture provides immediate feedback upon upload, while real-time socket communication keeps you updated on analysis progress from start to finish.",
            },
            {
                icon: ShieldCheck,
                title: "Comprehensive Forensic Reports",
                description:
                    "Go beyond a simple 'real' or 'fake' verdict. Dive deep with frame-by-frame confidence charts, processing environment metadata, and downloadable PDF reports for thorough documentation.",
            },
        ],
        models: [
            {
                name: "SIGLIP-LSTM V3",
                version: "3.0.0",
                description:
                    "Our most advanced and accurate model, combining a powerful vision encoder with temporal analysis for high-stakes forensic investigation.",
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
        ],
        team: [
            {
                name: "Kandarp Gajjar",
                role: "Full-Stack & AI Architect",
                avatarUrl:
                    "https://media.licdn.com/dms/image/v2/D4D03AQHv4z319ikShg/profile-displayphoto-shrink_800_800/B4DZX7n4OZG4Ag-/0/1743683290976?e=1758153600&v=beta&t=B0h6PCYUkUvWGFldVDOSEfeua5mSi_QnW__m5Q5ggzo",
                socials: {
                    github: "https://github.com/slantie",
                    linkedin: "https://www.linkedin.com/in/kandarpgajjar/",
                    twitter: "#",
                },
            },
            {
                name: "Oum Gadani",
                role: "AI/ML Engineer",
                avatarUrl:
                    "https://media.licdn.com/dms/image/v2/D4D03AQHkkFY5CkWVdA/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1728641354282?e=1758153600&v=beta&t=5Eq4gizCCfTPbm_3a1LXi8gtbz0GQ5hc_ds03oSBupk",
                socials: {
                    github: "#",
                    linkedin: "https://www.linkedin.com/in/oumgadani/",
                    twitter: "#",
                },
            },
            {
                name: "Raj Mathuria",
                role: "DevOps & Backend Specialist",
                avatarUrl:
                    "https://media.licdn.com/dms/image/v2/D4D03AQFl7WnWrj8HrA/profile-displayphoto-shrink_800_800/B4DZTxtlzvG4Ac-/0/1739222049898?e=1758153600&v=beta&t=Yz7KyqyNKHFIb29J_z1DiMiqfBS6Ur8nD5_h4bJFtHM",
                socials: {
                    github: "#",
                    linkedin:
                        "https://www.linkedin.com/in/raj-mathuria-98a710283/",
                    twitter: "#",
                },
            },
            {
                name: "Vishwajit Sarnobat",
                role: "UX & Frontend Developer",
                avatarUrl:
                    "https://media.licdn.com/dms/image/v2/D4D03AQHFDO-Uz-mqaw/profile-displayphoto-shrink_800_800/B4DZTpmgkNGkAc-/0/1739085975172?e=1758153600&v=beta&t=hqOvd7WWhCSleOzcmPgEbMhs784XXLgozmp6feAjDP8",
                socials: {
                    github: "https://github.com/vishwajitsarnobat",
                    linkedin: "https://www.linkedin.com/in/vishwajitsarnobat/",
                    twitter: "#",
                },
            },
        ],
    };

    const techStack = [
        "React",
        "Vite",
        "Tailwind CSS",
        "TanStack Query",
        "Recharts",
        "Node.js",
        "Express",
        "Prisma",
        "PostgreSQL",
        "Socket.IO",
        "FastAPI",
        "PyTorch",
        "OpenCV",
        "Dlib",
        "Redis",
        "Docker",
        "Cloudinary",
        "GitHub Actions",
    ];

    const stats = [
        { icon: Users, number: "75%", label: "Detection Accuracy" },
        { icon: Clock, number: "<30s", label: "Average Analysis Time" },
        { icon: Star, number: "2", label: "AI Models" },
        // { icon: ShieldCheck, number: "24/7", label: "System Uptime" },
    ];

    return (
        <div>
            <div className="min-h-screen bg-light-muted-background dark:bg-dark-background text-light-text dark:text-dark-text transition-colors duration-300">
                {/* Hero Section */}
                <section className="pt-24 pb-20 px-4 sm:px-6 lg:px-8">
                    <div className="max-w-7xl mx-auto">
                        <div className="text-center">
                            <div className="inline-flex items-center bg-light-background dark:bg-dark-muted-background border border-light-secondary dark:border-dark-secondary rounded-full px-4 py-2 mb-8">
                                <span className="w-2 h-2 bg-primary-main rounded-full mr-2 animate-pulse"></span>
                                <span className="text-sm font-medium">
                                    Advanced AI-Powered Detection
                                </span>
                            </div>

                            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold mb-6">
                                <span className="block">
                                    {projectData.projectDetails.name}
                                </span>
                                <span className="block text-primary-main mt-2">
                                    {projectData.projectDetails.tagline}
                                </span>
                            </h1>

                            <p className="text-xl text-light-muted-text dark:text-dark-text max-w-3xl mx-auto mb-12 leading-relaxed">
                                {projectData.projectDetails.description}
                            </p>

                            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-16">
                                <a
                                    // go to /login page
                                    href="/auth?view=login"
                                >
                                    <button className="bg-primary-main hover:bg-primary-light text-white px-8 py-4 rounded-lg font-semibold flex items-center group transition-all transform hover:scale-105">
                                        <Play className="w-5 h-5 mr-2" />
                                        Start Analysis
                                        <ChevronRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
                                    </button>
                                </a>
                                <a
                                    href="https://deepwiki.com/zaptrixio-cyber/Drishtiksha"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                >
                                    <button className="border-2 border-light-secondary dark:border-dark-secondary px-8 py-4 rounded-lg font-semibold hover:bg-light-hover dark:hover:bg-dark-hover transition-colors">
                                        View Documentation
                                    </button>
                                </a>
                            </div>

                            {/* Hero Stats */}
                            <div className="grid grid-cols-2 md:grid-cols-3 max-w-4xl mx-auto">
                                {stats.map((stat, index) => (
                                    <div key={index} className="text-center">
                                        <div className="inline-flex items-center justify-center w-12 h-12 bg-primary-main/10 rounded-lg mb-3">
                                            <stat.icon className="w-6 h-6 text-primary-main" />
                                        </div>
                                        <div className="text-2xl font-bold text-primary-main mb-1">
                                            {stat.number}
                                        </div>
                                        <div className="text-sm text-light-muted-text dark:text-dark-tertiary">
                                            {stat.label}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </section>
                {/* Features Section */}
                <section
                    id="features"
                    className="py-20 px-4 sm:px-6 lg:px-8 bg-light-background dark:bg-dark-muted-background"
                >
                    <div className="max-w-7xl mx-auto">
                        <div className="text-center mb-16">
                            <h2 className="text-4xl font-bold mb-4">
                                Cutting-Edge Features
                            </h2>
                            <p className="text-xl text-light-muted-text dark:text-dark-text max-w-2xl mx-auto">
                                Powered by advanced AI models and modern
                                architecture for unparalleled detection accuracy
                            </p>
                        </div>

                        <div className="grid md:grid-cols-3 gap-8">
                            {projectData.features.map((feature, index) => (
                                <div key={index} className="group">
                                    <div className="bg-light-muted-background dark:bg-dark-background rounded-2xl p-8 h-full border border-light-secondary dark:border-dark-secondary hover:border-primary-main dark:hover:border-primary-main transition-all duration-300 hover:shadow-xl">
                                        <div className="inline-flex items-center justify-center w-16 h-16 bg-primary-main/10 rounded-xl mb-6 group-hover:bg-primary-main/20 transition-colors">
                                            <feature.icon className="w-8 h-8 text-primary-main" />
                                        </div>
                                        <h3 className="text-xl font-semibold mb-4">
                                            {feature.title}
                                        </h3>
                                        <p className="text-light-muted-text dark:text-dark-text leading-relaxed">
                                            {feature.description}
                                        </p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>
                <section id="models" className="py-20 px-4 sm:px-6 lg:px-8">
                    <div className="max-w-7xl mx-auto">
                        <div className="text-center mb-16">
                            <h2 className="text-4xl font-bold mb-4">
                                AI Model Arsenal
                            </h2>
                            <p className="text-xl text-light-muted-text dark:text-dark-text max-w-2xl mx-auto">
                                Multiple specialized models working together for
                                comprehensive detection
                            </p>
                        </div>

                        <div className="space-y-6">
                            {/*
                                              map model names to available icons (fall back to CheckCircle)
                                              Uses imported lucide-react icons: BrainCircuit, Zap, ShieldCheck, CheckCircle
                                            */}
                            {projectData.models.map((model, index) => {
                                const modelIconMap = {
                                    "SIGLIP-LSTM V3": BrainCircuit,
                                    "Color Cues LSTM V1": Zap,
                                    "SIGLIP-LSTM V1 (Legacy)": ShieldCheck,
                                };
                                const Icon =
                                    modelIconMap[model.name] || CheckCircle;

                                return (
                                    <div
                                        key={index}
                                        className="bg-light-background dark:bg-dark-muted-background rounded-2xl p-8 border border-light-secondary dark:border-dark-secondary hover:border-primary-main dark:hover:border-primary-main transition-all duration-300"
                                    >
                                        <div className="flex flex-col lg:flex-row lg:items-center justify-between">
                                            <div className="flex-1">
                                                <div className="flex items-center mb-4">
                                                    <div className="inline-flex items-center justify-center w-12 h-12 bg-primary-main/10 rounded-full mr-4">
                                                        <Icon className="w-6 h-6 text-primary-main" />
                                                    </div>

                                                    <h3 className="text-2xl font-bold mr-4">
                                                        {model.name}
                                                    </h3>
                                                    <span className="bg-primary-main/10 text-primary-main px-3 py-1 rounded-full text-sm font-medium">
                                                        v{model.version}
                                                    </span>
                                                </div>
                                                <p className="text-light-muted-text dark:text-dark-text mb-4 leading-relaxed">
                                                    {model.description}
                                                </p>
                                            </div>
                                            <div className="lg:ml-8">
                                                <div className="inline-flex items-center bg-primary-main/10 text-primary-main px-4 py-2 rounded-lg font-medium">
                                                    <CheckCircle className="w-4 h-4 mr-2" />
                                                    {model.specialty}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </section>
                {/* Technology Stack */}
                <section
                    id="tech"
                    className="py-20 px-4 sm:px-6 lg:px-8 bg-light-background dark:bg-dark-muted-background"
                >
                    <div className="max-w-7xl mx-auto">
                        <div className="text-center mb-16">
                            <h2 className="text-4xl font-bold mb-4">
                                Technology Stack
                            </h2>
                            <p className="text-xl text-light-muted-text dark:text-dark-text max-w-2xl mx-auto">
                                Built with modern, scalable technologies for
                                optimal performance
                            </p>
                        </div>

                        {/* Marquee Container */}
                        <div className="relative overflow-hidden bg-light-muted-background dark:bg-dark-background rounded-2xl border border-light-secondary dark:border-dark-secondary py-8">
                            <div className="flex animate-marquee whitespace-nowrap">
                                {/* icon map for tech names (fall back to ArrowUp) */}
                                {techStack.map((tech, index) => {
                                    const techIconMap = {
                                        React: SiReact,
                                        Vite: SiVite,
                                        "Tailwind CSS": SiTailwindcss,
                                        "TanStack Query": SiReactquery,
                                        Recharts: ChartCandlestickIcon,
                                        "Node.js": SiNodedotjs,
                                        Express: SiExpress,
                                        Prisma: SiPrisma,
                                        PostgreSQL: SiPostgresql,
                                        "Socket.IO": SiSocketdotio,
                                        FastAPI: SiFastapi,
                                        PyTorch: SiPytorch,
                                        OpenCV: SiOpencv,
                                        Dlib: ScanFaceIcon,
                                        Redis: SiRedis,
                                        Docker: SiDocker,
                                        Cloudinary: SiCloudinary,
                                        "GitHub Actions": SiGithubactions,
                                    };
                                    const TechIcon =
                                        techIconMap[tech] || ArrowUp;

                                    return (
                                        <div
                                            key={`first-${index}`}
                                            className="inline-flex flex-col items-center mx-8"
                                        >
                                            <div className="flex flex-col items-center px-6 py-3">
                                                <TechIcon className="w-14 h-14 text-primary-main mb-2 hover:scale-105" />
                                                <span className="font-medium text-light-text dark:text-dark-text whitespace-nowrap">
                                                    {tech}
                                                </span>
                                            </div>
                                        </div>
                                    );
                                })}
                                {/* Duplicate set for seamless loop */}
                                {techStack.map((tech, index) => {
                                    const techIconMap = {
                                        React: SiReact,
                                        Vite: SiVite,
                                        "Tailwind CSS": SiTailwindcss,
                                        "TanStack Query": SiReactquery,
                                        Recharts: ChartCandlestickIcon,
                                        "Node.js": SiNodedotjs,
                                        Express: SiExpress,
                                        Prisma: SiPrisma,
                                        PostgreSQL: SiPostgresql,
                                        "Socket.IO": SiSocketdotio,
                                        FastAPI: SiFastapi,
                                        PyTorch: SiPytorch,
                                        OpenCV: SiOpencv,
                                        Dlib: ScanFaceIcon,
                                        Redis: SiRedis,
                                        Docker: SiDocker,
                                        Cloudinary: SiCloudinary,
                                        "GitHub Actions": SiGithubactions,
                                    };
                                    const TechIcon =
                                        techIconMap[tech] || ArrowUp;

                                    return (
                                        <div
                                            key={`first-${index}`}
                                            className="inline-flex flex-col items-center mx-8"
                                        >
                                            <div className="flex flex-col items-center px-6 py-3">
                                                <TechIcon className="w-14 h-14 text-primary-main mb-2 hover:scale-105" />
                                                <span className="font-medium text-light-text dark:text-dark-text whitespace-nowrap">
                                                    {tech}
                                                </span>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>

                        {/* Additional info */}
                        <div className="text-center mt-8">
                            <p className="text-light-muted-text dark:text-dark-tertiary text-sm">
                                Continuously updated with the latest
                                industry-standard technologies
                            </p>
                        </div>
                    </div>
                </section>
                {/* Team Section */}
                <section id="team" className="py-20 px-4 sm:px-6 lg:px-8">
                    <div className="max-w-7xl mx-auto">
                        <div className="text-center mb-16">
                            <h2 className="text-4xl font-bold mb-4">
                                Meet Our Team
                            </h2>
                            <p className="text-xl text-light-muted-text dark:text-dark-text max-w-2xl mx-auto">
                                Passionate experts dedicated to fighting
                                deepfake technology
                            </p>
                        </div>

                        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
                            {projectData.team.map((member, index) => (
                                <div key={index} className="text-center group">
                                    <div className="bg-light-background dark:bg-dark-muted-background rounded-2xl p-6 border border-light-secondary dark:border-dark-secondary hover:border-primary-main dark:hover:border-primary-main transition-all duration-300 hover:shadow-lg">
                                        <img
                                            src={member.avatarUrl}
                                            alt={member.name}
                                            className="w-32 h-32 rounded-full mx-auto mb-4 border-4 border-primary-main/20 group-hover:border-primary-main/40 transition-colors"
                                        />
                                        <h3 className="font-semibold text-lg mb-2">
                                            {member.name}
                                        </h3>
                                        <p className="text-primary-main text-sm font-medium mb-4">
                                            {member.role}
                                        </p>
                                        <div className="flex justify-center space-x-3">
                                            {member.socials.github &&
                                                member.socials.github !==
                                                    "#" && (
                                                    <a
                                                        href={
                                                            member.socials
                                                                .github
                                                        }
                                                        className="text-light-muted-text dark:text-dark-tertiary hover:text-primary-main transition-colors"
                                                    >
                                                        <Github className="w-5 h-5" />
                                                    </a>
                                                )}
                                            {member.socials.linkedin &&
                                                member.socials.linkedin !==
                                                    "#" && (
                                                    <a
                                                        href={
                                                            member.socials
                                                                .linkedin
                                                        }
                                                        className="text-light-muted-text dark:text-dark-tertiary hover:text-primary-main transition-colors"
                                                    >
                                                        <Linkedin className="w-5 h-5" />
                                                    </a>
                                                )}
                                            {member.socials.twitter &&
                                                member.socials.twitter !==
                                                    "#" && (
                                                    <a
                                                        href={
                                                            member.socials
                                                                .twitter
                                                        }
                                                        className="text-light-muted-text dark:text-dark-tertiary hover:text-primary-main transition-colors"
                                                    >
                                                        <Twitter className="w-5 h-5" />
                                                    </a>
                                                )}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>
            </div>
        </div>
    );
};

export default Home;
