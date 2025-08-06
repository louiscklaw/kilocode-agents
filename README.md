# kilocode Subagents Collection

## Credits (intended to be in front of anything)

this project shamelessly fork from [Claude Code Subagents Collection](https://github.com/wshobson/agents)

## Overview

Sorry but this project is still work-in-progress

A comprehensive collection of specialized AI subagents for [Claude Code](https://docs.anthropic.com/en/docs/claude-code), designed to enhance development workflows with domain-specific expertise.

This repository contains 50 specialized subagents that extend Claude Code's capabilities. Each subagent is an expert in a specific domain, automatically invoked based on context or explicitly called when needed. All agents are configured with specific Claude models based on task complexity for optimal performance and cost-effectiveness.

## Available Subagents

### Development & Architecture
- **[backend-architect](yaml/backend-architect.yaml)** - Design RESTful APIs, microservice boundaries, and database schemas
- **[frontend-developer](yaml/frontend-developer.yaml)** - Build React components, implement responsive layouts, and handle client-side state management
- **[ui-ux-designer](yaml/ui-ux-designer.yaml)** - Create interface designs, wireframes, and design systems
- **[mobile-developer](yaml/mobile-developer.yaml)** - Develop React Native or Flutter apps with native integrations
- **[graphql-architect](yaml/graphql-architect.yaml)** - Design GraphQL schemas, resolvers, and federation
- **[architect-reviewer](yaml/architect-review.yaml)** - Reviews code changes for architectural consistency and patterns

### Language Specialists
- **[python-pro](yaml/python-pro.yaml)** - Write idiomatic Python code with advanced features and optimizations
- **[golang-pro](yaml/golang-pro.yaml)** - Write idiomatic Go code with goroutines, channels, and interfaces
- **[rust-pro](yaml/rust-pro.yaml)** - Write idiomatic Rust with ownership patterns, lifetimes, and trait implementations
- **[c-pro](yaml/c-pro.yaml)** - Write efficient C code with proper memory management and system calls
- **[cpp-pro](yaml/cpp-pro.yaml)** - Write idiomatic C++ code with modern features, RAII, smart pointers, and STL algorithms
- **[javascript-pro](yaml/javascript-pro.yaml)** - Master modern JavaScript with ES6+, async patterns, and Node.js APIs
- **[typescript-pro](yaml/typescript-pro.yaml)** - Master TypeScript with advanced types, generics, and strict type safety
- **[php-pro](yaml/php-pro.yaml)** - Write idiomatic PHP code with modern features and performance optimizations
- **[java-pro](yaml/java-pro.yaml)** - Master modern Java with streams, concurrency, and JVM optimization
- **[ios-developer](yaml/ios-developer.yaml)** - Develop native iOS applications with Swift/SwiftUI
- **[sql-pro](yaml/sql-pro.yaml)** - Write complex SQL queries, optimize execution plans, and design normalized schemas

### Infrastructure & Operations
- **[devops-troubleshooter](yaml/devops-troubleshooter.yaml)** - Debug production issues, analyze logs, and fix deployment failures
- **[deployment-engineer](yaml/deployment-engineer.yaml)** - Configure CI/CD pipelines, Docker containers, and cloud deployments
- **[cloud-architect](yaml/cloud-architect.yaml)** - Design AWS/Azure/GCP infrastructure and optimize cloud costs
- **[database-optimizer](yaml/database-optimizer.yaml)** - Optimize SQL queries, design efficient indexes, and handle database migrations
- **[database-admin](yaml/database-admin.yaml)** - Manage database operations, backups, replication, and monitoring
- **[terraform-specialist](yaml/terraform-specialist.yaml)** - Write advanced Terraform modules, manage state files, and implement IaC best practices
- **[incident-responder](yaml/incident-responder.yaml)** - Handles production incidents with urgency and precision
- **[network-engineer](yaml/network-engineer.yaml)** - Debug network connectivity, configure load balancers, and analyze traffic patterns
- **[dx-optimizer](yaml/dx-optimizer.yaml)** - Developer Experience specialist that improves tooling, setup, and workflows

### Quality & Security
- **[code-reviewer](yaml/code-reviewer.yaml)** - Expert code review with deep configuration security focus and production reliability
- **[security-auditor](yaml/security-auditor.yaml)** - Review code for vulnerabilities and ensure OWASP compliance
- **[test-automator](yaml/test-automator.yaml)** - Create comprehensive test suites with unit, integration, and e2e tests
- **[performance-engineer](yaml/performance-engineer.yaml)** - Profile applications, optimize bottlenecks, and implement caching strategies
- **[debugger](yaml/debugger.yaml)** - Debugging specialist for errors, test failures, and unexpected behavior
- **[error-detective](yaml/error-detective.yaml)** - Search logs and codebases for error patterns, stack traces, and anomalies
- **[search-specialist](yaml/search-specialist.yaml)** - Expert web researcher using advanced search techniques and synthesis

### Data & AI
- **[data-scientist](yaml/data-scientist.yaml)** - Data analysis expert for SQL queries, BigQuery operations, and data insights
- **[data-engineer](yaml/data-engineer.yaml)** - Build ETL pipelines, data warehouses, and streaming architectures
- **[ai-engineer](yaml/ai-engineer.yaml)** - Build LLM applications, RAG systems, and prompt pipelines
- **[ml-engineer](yaml/ml-engineer.yaml)** - Implement ML pipelines, model serving, and feature engineering
- **[mlops-engineer](yaml/mlops-engineer.yaml)** - Build ML pipelines, experiment tracking, and model registries
- **[prompt-engineer](yaml/prompt-engineer.yaml)** - Optimizes prompts for LLMs and AI systems

### Specialized Domains
- **[api-documenter](yaml/api-documenter.yaml)** - Create OpenAPI/Swagger specs and write developer documentation
- **[payment-integration](yaml/payment-integration.yaml)** - Integrate Stripe, PayPal, and payment processors
- **[quant-analyst](yaml/quant-analyst.yaml)** - Build financial models, backtest trading strategies, and analyze market data
- **[risk-manager](yaml/risk-manager.yaml)** - Monitor portfolio risk, R-multiples, and position limits
- **[legacy-modernizer](yaml/legacy-modernizer.yaml)** - Refactor legacy codebases and implement gradual modernization
- **[context-manager](yaml/context-manager.yaml)** - Manages context across multiple agents and long-running tasks

### Business & Marketing
- **[business-analyst](yaml/business-analyst.yaml)** - Analyze metrics, create reports, and track KPIs
- **[content-marketer](yaml/content-marketer.yaml)** - Write blog posts, social media content, and email newsletters
- **[sales-automator](yaml/sales-automator.yaml)** - Draft cold emails, follow-ups, and proposal templates
- **[customer-support](yaml/customer-support.yaml)** - Handle support tickets, FAQ responses, and customer emails
- **[legal-advisor](yaml/legal-advisor.yaml)** - Draft privacy policies, terms of service, disclaimers, and legal notices

## Model Assignments

All 50 subagents are configured with specific Claude models based on task complexity:

### 🚀 Haiku (Fast & Cost-Effective) - 8 agents
**Model:** `haiku`
- `data-scientist` - SQL queries and data analysis
- `api-documenter` - OpenAPI/Swagger documentation
- `business-analyst` - Metrics and KPI tracking
- `content-marketer` - Blog posts and social media
- `customer-support` - Support tickets and FAQs
- `sales-automator` - Cold emails and proposals
- `search-specialist` - Web research and information gathering
- `legal-advisor` - Privacy policies and compliance documents

### ⚡ Sonnet (Balanced Performance) - 31 agents
**Model:** `sonnet`

**Development & Languages:**
- `python-pro` - Python development with advanced features
- `javascript-pro` - Modern JavaScript and Node.js
- `typescript-pro` - Advanced TypeScript with type systems
- `golang-pro` - Go concurrency and idiomatic patterns
- `rust-pro` - Rust memory safety and systems programming
- `c-pro` - C programming and embedded systems
- `cpp-pro` - Modern C++ with STL and templates
- `php-pro` - Modern PHP with advanced features
- `java-pro` - Modern Java with streams and concurrency
- `ios-developer` - Native iOS development with Swift/SwiftUI
- `frontend-developer` - React components and UI
- `ui-ux-designer` - Interface design and wireframes
- `backend-architect` - API design and microservices
- `mobile-developer` - React Native/Flutter apps
- `sql-pro` - Complex SQL optimization
- `graphql-architect` - GraphQL schemas and resolvers

**Infrastructure & Operations:**
- `devops-troubleshooter` - Production debugging
- `deployment-engineer` - CI/CD pipelines
- `database-optimizer` - Query optimization
- `database-admin` - Database operations
- `terraform-specialist` - Infrastructure as Code
- `network-engineer` - Network configuration
- `dx-optimizer` - Developer experience
- `data-engineer` - ETL pipelines

**Quality & Support:**
- `test-automator` - Test suite creation
- `code-reviewer` - Code quality analysis
- `debugger` - Error investigation
- `error-detective` - Log analysis
- `ml-engineer` - ML model deployment
- `legacy-modernizer` - Framework migrations
- `payment-integration` - Payment processing

### 🧠 Opus (Maximum Capability) - 11 agents
**Model:** `opus`
- `ai-engineer` - LLM applications and RAG systems
- `security-auditor` - Vulnerability analysis
- `performance-engineer` - Application optimization
- `incident-responder` - Production incident handling
- `mlops-engineer` - ML infrastructure
- `architect-reviewer` - Architectural consistency
- `cloud-architect` - Cloud infrastructure design
- `prompt-engineer` - LLM prompt optimization
- `context-manager` - Multi-agent coordination
- `quant-analyst` - Financial modeling
- `risk-manager` - Portfolio risk management

## Installation

These subagents are automatically available when placed in `~/.claude/agents/` directory.

```bash
cd ~/.claude
git clone https://github.com/wshobson/agents.git
```

## Usage

### Automatic Invocation
Claude Code will automatically delegate to the appropriate subagent based on the task context and the subagent's description.

### Explicit Invocation
Mention the subagent by name in your request:
```
"Use the code-reviewer to check my recent changes"
"Have the security-auditor scan for vulnerabilities"
"Get the performance-engineer to optimize this bottleneck"
```

## Usage Examples

### Single Agent Tasks
```bash
# Code quality and review
"Use code-reviewer to analyze this component for best practices"
"Have code-reviewer scrutinize these configuration changes"
"Have security-auditor check for OWASP compliance issues"

# Development tasks
"Get backend-architect to design a user authentication API"
"Use frontend-developer to create a responsive dashboard layout"

# Infrastructure and operations
"Have devops-troubleshooter analyze these production logs"
"Use cloud-architect to design a scalable AWS architecture"
"Get network-engineer to debug SSL certificate issues"
"Use database-admin to set up backup and replication"

# Data and AI
"Get data-scientist to analyze this customer behavior dataset"
"Use ai-engineer to build a RAG system for document search"
"Have mlops-engineer set up MLflow experiment tracking"

# Business and marketing
"Have business-analyst create investor deck with growth metrics"
"Use content-marketer to write SEO-optimized blog post"
"Get sales-automator to create cold email sequence"
"Have customer-support draft FAQ documentation"
```

### Multi-Agent Workflows

These subagents work together seamlessly, and for more complex orchestrations, you can use the **[Claude Code Commands](https://github.com/wshobson/commands)** collection which provides 52 pre-built slash commands that leverage these subagents in sophisticated workflows.

```bash
# Feature development workflow
"Implement user authentication feature"
# Automatically uses: backend-architect → frontend-developer → test-automator → security-auditor

# Performance optimization workflow
"Optimize the checkout process performance"
# Automatically uses: performance-engineer → database-optimizer → frontend-developer

# Production incident workflow
"Debug high memory usage in production"
# Automatically uses: incident-responder → devops-troubleshooter → error-detective → performance-engineer

# Network connectivity workflow
"Fix intermittent API timeouts"
# Automatically uses: network-engineer → devops-troubleshooter → performance-engineer

# Database maintenance workflow
"Set up disaster recovery for production database"
# Automatically uses: database-admin → database-optimizer → incident-responder

# ML pipeline workflow
"Build end-to-end ML pipeline with monitoring"
# Automatically uses: mlops-engineer → ml-engineer → data-engineer → performance-engineer

# Product launch workflow
"Launch new feature with marketing campaign"
# Automatically uses: business-analyst → content-marketer → sales-automator → customer-support
```

### Advanced Workflows with Slash Commands

For more sophisticated multi-subagent orchestration, use the companion [Commands repository](https://github.com/wshobson/commands):

```bash
# Complex feature development (8+ subagents)
/full-stack-feature Build user dashboard with real-time analytics

# Production incident response (5+ subagents)
/incident-response Database connection pool exhausted

# ML infrastructure setup (6+ subagents)
/ml-pipeline Create recommendation engine with A/B testing

# Security-focused implementation (7+ subagents)
/security-hardening Implement OAuth2 with zero-trust architecture
```

## Subagent Format

Each subagent follows this structure:
```yaml
---
name: subagent-name
description: When this subagent should be invoked
model: haiku  # Optional - specify which model to use (haiku/sonnet/opus)
tools: tool1, tool2  # Optional - defaults to all tools
---

System prompt defining the subagent's role and capabilities
```

### Model Configuration

As of Claude Code v1.0.64, subagents can specify which Claude model they should use. This allows for cost-effective task delegation based on complexity:

- **Low Complexity (Haiku)**: Simple tasks like basic data analysis, documentation generation, and standard responses
- **Medium Complexity (Sonnet)**: Development tasks, code review, testing, and standard engineering work
- **High Complexity (Opus)**: Critical tasks like security auditing, architecture review, incident response, and AI/ML engineering

Available models (using simplified naming as of Claude Code v1.0.64):
- `haiku` - Fast and cost-effective for simple tasks
- `sonnet` - Balanced performance for most development work
- `opus` - Most capable for complex analysis and critical tasks

If no model is specified, the subagent will use the system's default model.

Note: All subagent files are now in YAML format (.yaml) instead of Markdown (.md)

## Agent Orchestration Patterns

Claude Code automatically coordinates agents using these common patterns:

### Sequential Workflows
```
User Request → Agent A → Agent B → Agent C → Result

Example: "Build a new API feature"
backend-architect → frontend-developer → test-automator → security-auditor
```

### Parallel Execution
```
User Request → Agent A + Agent B (simultaneously) → Merge Results

Example: "Optimize application performance"
performance-engineer + database-optimizer → Combined recommendations
```

### Conditional Branching
```
User Request → Analysis → Route to appropriate specialist

Example: "Fix this bug"
debugger (analyzes) → Routes to: backend-architect OR frontend-developer OR devops-troubleshooter
```

### Review & Validation
```
Primary Agent → Review Agent → Final Result

Example: "Implement payment processing"
payment-integration → security-auditor → Validated implementation
```

## When to Use Which Agent

### 🏗️ Planning & Architecture
- **backend-architect**: API design, database schemas, system architecture
- **frontend-developer**: UI/UX planning, component architecture
- **ui-ux-designer**: Interface design, wireframes, design systems, user research
- **cloud-architect**: Infrastructure design, scalability planning

### 🔧 Implementation & Development
- **python-pro**: Python-specific development tasks
- **golang-pro**: Go-specific development tasks
- **rust-pro**: Rust-specific development, memory safety, systems programming
- **c-pro**: C programming, embedded systems, performance-critical code
- **javascript-pro**: Modern JavaScript, async patterns, Node.js/browser code
- **typescript-pro**: Advanced TypeScript, generics, type inference, enterprise patterns
- **java-pro**: Modern Java development, streams, concurrency, Spring Boot
- **ios-developer**: Native iOS development with Swift/SwiftUI
- **sql-pro**: Database queries, schema design, query optimization
- **mobile-developer**: React Native/Flutter development

### 🛠️ Operations & Maintenance
- **devops-troubleshooter**: Production issues, deployment problems
- **incident-responder**: Critical outages requiring immediate response
- **database-optimizer**: Query performance, indexing strategies
- **database-admin**: Backup strategies, replication, user management, disaster recovery
- **terraform-specialist**: Infrastructure as Code, Terraform modules, state management
- **network-engineer**: Network connectivity, load balancers, SSL/TLS, DNS debugging

### 📊 Analysis & Optimization
- **performance-engineer**: Application bottlenecks, optimization
- **security-auditor**: Vulnerability scanning, compliance checks
- **data-scientist**: Data analysis, insights, reporting
- **mlops-engineer**: ML infrastructure, experiment tracking, model registries, pipeline automation

### 🧪 Quality Assurance
- **code-reviewer**: Code quality, configuration security, production reliability
- **test-automator**: Test strategy, test suite creation
- **debugger**: Bug investigation, error resolution
- **error-detective**: Log analysis, error pattern recognition, root cause analysis
- **search-specialist**: Deep web research, competitive analysis, fact-checking

### 💼 Business & Strategy
- **business-analyst**: KPIs, revenue models, growth projections, investor metrics
- **risk-manager**: Portfolio risk, hedging strategies, R-multiples, position sizing
- **content-marketer**: SEO content, blog posts, social media, email campaigns
- **sales-automator**: Cold emails, follow-ups, proposals, lead nurturing
- **customer-support**: Support tickets, FAQs, help documentation, troubleshooting
- **legal-advisor** - Draft privacy policies, terms of service, disclaimers, and legal notices

## Best Practices

### 🎯 Task Delegation
1. **Let Claude Code delegate automatically** - The main agent analyzes context and selects optimal agents
2. **Be specific about requirements** - Include constraints, tech stack, and quality requirements
3. **Trust agent expertise** - Each agent is optimized for their domain

### 🔄 Multi-Agent Workflows
4. **Start with high-level requests** - Let agents coordinate complex multi-step tasks
5. **Provide context between agents** - Ensure agents have necessary background information
6. **Review integration points** - Check how different agents' outputs work together

### 🎛️ Explicit Control
7. **Use explicit invocation for specific needs** - When you want a particular expert's perspective
8. **Combine multiple agents strategically** - Different specialists can validate each other's work
9. **Request specific review patterns** - "Have security-auditor review backend-architect's API design"

### 📈 Optimization
10. **Monitor agent effectiveness** - Learn which agents work best for your use cases
11. **Iterate on complex tasks** - Use agent feedback to refine requirements
12. **Leverage agent strengths** - Match task complexity to agent capabilities

## Contributing

To add a new subagent:
1. Create a new `.yaml` file following the format above
2. Use lowercase, hyphen-separated names
3. Write clear descriptions for when the subagent should be used
4. Include specific instructions in the system prompt

## Troubleshooting

### Common Issues

**Agent not being invoked automatically:**
- Ensure your request clearly indicates the domain (e.g., "performance issue" → performance-engineer)
- Be specific about the task type (e.g., "review code" → code-reviewer)

**Unexpected agent selection:**
- Provide more context about your tech stack and requirements
- Use explicit invocation if you need a specific agent

**Multiple agents producing conflicting advice:**
- This is normal - different specialists may have different priorities
- Ask for clarification: "Reconcile the recommendations from security-auditor and performance-engineer"

**Agent seems to lack context:**
- Provide background information in your request
- Reference previous conversations or established patterns

### Getting Help

If agents aren't working as expected:
1. Check agent descriptions in their individual files
2. Try more specific language in your requests
3. Use explicit invocation to test specific agents
4. Provide more context about your project and goals

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Learn More

- [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
- [Subagents Documentation](https://docs.anthropic.com/en/docs/claude-code/sub-agents)
- [Claude Code GitHub](https://github.com/anthropics/claude-code)
