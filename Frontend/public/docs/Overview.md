# Drishtiksha: A Real-Time Deepfake Detection Platform

**Drishtiksha** is a comprehensive, enterprise-grade platform designed for the detection and forensic analysis of deepfakes in digital media. It is architected as a robust, scalable system of microservices, combining a modern web interface, a powerful backend orchestrator, and an extensible AI inference engine.

The platform's core mandate is to provide a seamless, end-to-end workflow for users to upload media (video, audio, and images), receive real-time feedback on the analysis progress, and explore detailed, multi-model forensic reports. It is designed from the ground up to handle computationally expensive, long-running AI tasks in a non-blocking, asynchronous manner, ensuring a responsive and reliable user experience.

### Key Features

*   **Microservice Architecture:** A decoupled system with a React frontend, Node.js backend, and Python ML server for independent scaling and maintenance.
*   **Asynchronous Job Queueing:** Utilizes **BullMQ** and **Redis** to manage long-running analysis tasks, ensuring the API remains highly responsive.
*   **Real-Time Progress Updates:** A **WebSocket** and **Redis Pub/Sub** system provides granular, real-time feedback to the user on analysis progress.
*   **Extensible AI Core:** The Python server features an automated **Model Registry** that dynamically discovers and loads any compatible AI model, making the system future-proof.
*   **Multi-Modal Analysis:** Supports video, audio, and image analysis with a suite of specialized, state-of-the-art AI models.
*   **Rich Data Visualization:** The frontend presents complex forensic data through interactive charts, graphs, and temporal analysis timelines.
*   **Multiple Client Interfaces:** The platform can be accessed via a responsive **web application**, a powerful **Command-Line Interface (CLI)**, and a convenient **Browser Extension**.
*   **Containerized & DevOps-Ready:** The entire platform is fully containerized with **Docker** and orchestrated with Docker Compose for consistent, one-command deployment.

---

## System Architecture

The Drishtiksha platform is composed of three primary services that communicate via REST APIs, WebSockets, and a Redis message broker. This decoupled architecture ensures scalability, resilience, and maintainability.


```text
         ┌───────────────────────────┐      ┌───────────────────────────┐
         │ Browser Extension Client  │      │   React Web Application   │
         └─────────────┬─────────────┘      └─────────────┬─────────────┘
                       │                                  │
                       │ REST API (for analysis)          │ REST API & WebSockets
                       │                                  │
         ┌─────────────▼──────────────────────────────────▼───────────────────┐
         │                          Backend (Node.js)                         │
         │       ─ API Gateway, Auth, Job Orchestration, Real-Time ─          │
         ├──────────────────────────────┬──────────────────────────────┬──────┘
         │                              │                              │      
         │       BullMQ Jobs            │      Redis Pub/Sub           │ HTTP 
         │                              │                              │ REST 
         │                              │                              │ API  
         │                              │                              │      
┌────────▼────────┐          ┌──────────▼──────────┐          ┌────────▼────────┐
│  PostgreSQL DB  │          │    Redis Server     │◄────────►│   ML Server     │
│ (Users, Media,  │          │ (Queue & Pub/Sub)   │          │   (Python)      │
│  Analysis Data) │          └─────────────────────┘          │ (AI Inference)  │
└─────────────────┘                                           └─────────────────┘

```

---

## Technology Stack

| Module | Category | Technologies |
| :--- | :--- | :--- |
| **Frontend** | UI/UX & State | React 19, Vite, Tailwind CSS, TanStack React Query, Socket.IO, Recharts |
| **Backend** | API & Orchestration | Node.js, Express.js, Prisma, PostgreSQL, BullMQ, Redis, Socket.IO |
| **ML Server** | AI & Inference | Python, FastAPI, PyTorch, Librosa, OpenCV, Timm, Transformers |
| **Browser Extension** | Integration | Manifest V3 APIs, JavaScript, HTML/CSS |
| **Infrastructure** | DevOps | Docker, Docker Compose, Nginx |

---

## Modules Overview

The project is divided into four primary, independent modules:

### 1. **Backend** (Node.js) - *The Orchestrator*

The central nervous system of the platform. It handles user authentication, provides the main REST API, manages the asynchronous job queue, and pushes real-time updates to the frontend. It orchestrates the entire analysis workflow without performing any heavy computation itself.

### 2. **Frontend** (React) - *The User Interface*

A sophisticated single-page application that provides a rich, responsive, and real-time user experience. It allows users to manage their media, upload new files, and explore detailed, interactive visualizations of the analysis results.

### 3. **Server** (Python) - *The AI Core*

A high-performance, specialized microservice built with FastAPI. It hosts the suite of AI models and exposes them via a secure REST API. Its sole purpose is to perform deepfake detection inference, accepting a media file and returning a structured JSON report.

### 4. **Browser Extension** - *The Productivity Tool*

A lightweight Chrome extension that integrates the Drishtiksha service directly into the user's browsing experience. It allows users to right-click any media on any website and send it for analysis with a single click, using their existing web application session for authentication.

---

## Quick Start (Docker)

The entire Drishtiksha platform is containerized and can be run with a single command. This is the recommended method for both development and deployment.

### 1. Prerequisites

*   **Docker & Docker Compose** installed on your system.
*   **Git** for cloning the repository.
*   **Model Weights:** You must manually download the required `.pth` and `.dat` model files and place them in the `Server/models/` directory.

### 2. Clone the Repository

```bash
git clone https://github.com/slantie/drishtiksha.git
cd drishtiksha
```

### 3. Configure the Environment

Create a single `.env` file in the project root by copying the provided Docker example.

```bash
cp .env.docker.example .env
```

Open the `.env` file and **change the default passwords and secret keys**. The network URLs are pre-configured to work within the Docker Compose environment.

### 4. Build and Run the Platform

Use the provided Docker Compose command to build all images and start all services (Frontend, Backend, ML Server, Database, Redis) in the background.

```bash
docker-compose -f docker-compose.yml -f docker-compose.local.yml up --build -d
```

### 5. Access the Application

Once all containers are up and running, you can access the platform:

*   **Frontend Web Application:** [http://localhost:5173](http://localhost:5173)
*   **Backend API Health:** [http://localhost:3000](http://localhost:3000)
*   **ML Server Health:** [http://localhost:8000](http://localhost:8000)
*   **Prisma Studio (Database GUI):** Run `docker-compose -f docker-compose.yml -f docker-compose.local.yml --profile studio up -d` and access [http://localhost:5555](http://localhost:5555).

---

## Project Structure

```text
.
├── Backend/          # Node.js API, Worker, and Orchestration Layer
├── BrowserExtension/ # Chrome Extension for in-browser analysis
├── Frontend/         # React Single-Page Application (UI/UX)
├── Server/           # Python FastAPI ML Inference Server (AI Core)
│
├── docker-compose.yml        # Base Docker Compose configuration
├── docker-compose.local.yml  # Local development overrides (e.g., database)
└── .env.docker.example       # Template for Docker environment configuration
```

---

## Vision & Conclusion

Drishtiksha is more than a deepfake detection tool; it is a complete, production-ready platform built on modern software engineering principles. Its microservice architecture ensures scalability and maintainability, while its asynchronous, event-driven nature provides a seamless user experience for complex computational tasks. The project serves as a powerful demonstration of how to integrate cutting-edge AI into a robust and user-friendly full-stack application.