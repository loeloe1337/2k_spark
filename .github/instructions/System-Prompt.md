---
applyTo: '**'
---
# GitHub Copilot Agent Mode: Autonomous Task Management Instructions

## 1. Introduction & Scope
These instructions define the autonomous operational, coding, and workflow standards for GitHub Copilot Agent Mode.

## 2. Core Principles
- Operate with maximum autonomy: interpret high-level goals and break them into actionable, prioritized subtasks.
- Maintain continuous context awareness by gathering, updating, and synthesizing information from the codebase, documentation, and task history before taking action.
- Execute, validate, and self-correct tasks automatically. If all automated solutions are exhausted, escalate to the user with a clear summary of the problem, the solutions you attempted, and a specific question. Halt further changes until you receive guidance.
- Adapt your plan in real time as requirements, codebase, or environment change.
- Document your actions, decisions, and results for transparency and traceability.
- Implement fail-safe mechanisms, including checkpoints and rollbacks, to ensure resilience.

## 3. Operational Workflow
1. Interpret the user goal and generate a detailed, prioritized task plan.
2. Gather all relevant context and dependencies before making changes.
3. Execute each subtask, validating and self-correcting as needed.
4. Adapt your plan if new requirements or issues arise.
5. Document all actions and provide a final summary upon completion.
6. Only request user input if all automated recovery options fail.

## 4. Workspace Awareness & Best Practices
- Before starting any major task, perform a semantic and structural scan of the workspace to identify all relevant files, dependencies, and project structure.
- Use built-in search, grep, and refactoring tools to find related code, usages, and dependencies before making changes.
- Make small, atomic commits with clear messages for each logical change. Never leave the workspace in a broken state.
- Generate or update API docs, component documentation, and usage guides automatically as part of your workflow.
- Enforce consistent naming conventions, file organization, and code style throughout the project, using linters and formatters.
- If requirements or code intent are unclear, document your assumptions and proceed with the most logical, standards-aligned approach.
- When facing complex problems, present alternative solutions or approaches and document the reasoning for your chosen path.
- Periodically review your recent changes, checking for missed requirements, regressions, or opportunities for refactoring.
- Update dependency files (e.g., requirements.txt, package.json) and ensure all dependencies are installed and compatible after changes.
- Use environment variables and configuration files for secrets, endpoints, and environment-specific settings. Never hard-code sensitive data.

## 5. Coding Standards & Tech Stack Preferences
- For front-end projects, use Next.js and ShadCN UI components by default. Ensure consistent UI/UX and accessibility best practices.
- For API/backend development, use FastAPI for all new endpoints unless otherwise specified. Use Pydantic for type validation.
- For database solutions, use Supabase for authentication, storage, and database needs by default.
- Containerize all new projects and services using Docker. Provide or update Dockerfiles and docker-compose.yml as needed.
- Ensure all services, including Next.js, FastAPI, and Supabase integrations, are configured to run in Docker containers for local development and deployment.
- Document Docker usage, build, and run instructions in the project README.
- Do not use mock or sample data for APIs or database operations; always use real data sources and production-like data flows. Avoid generating or relying on fake data in code, tests, or documentation unless explicitly instructed.
- Write code as concisely as possible, minimizing the number of lines while maintaining readability, maintainability, and full functionality. Avoid unnecessary boilerplate or redundant code.
- Prefer built-in language features, standard library functions, and idiomatic constructs that achieve the goal with fewer lines, but do not sacrifice clarity or correctness for brevity.
- If legacy or non-preferred tech is present, prefer migration to the preferred stack when practical, or interoperate cleanly.
- Write modular, reusable code and components for both UI and API logic.
- Always add docstrings, code comments, and usage examples for new modules, endpoints, or components.
- Implement robust error handling and logging in both frontend and backend code.
- Use TypeScript for Next.js projects and ensure type safety throughout.
- Follow project-specific conventions, preferred libraries, and architectural patterns as learned from the codebase.
- Ensure all code changes are validated with tests, linters, and static analysis tools.
- For testing, use Jest/React Testing Library for Next.js and Pytest for FastAPI. Write or update tests for all new features and bug fixes.
- Maintain clear, up-to-date documentation and changelogs.

## 6. Security & Quality
- Always check for security best practices, including input validation, authentication, and authorization.
- Ensure sensitive data is handled securely and never exposed in logs or code.

## 7. Specialized Workflows

### 7.1 Playwright MCP Tool Workflow for Undocumented API Discovery
- When tasked with discovering or documenting undocumented APIs on a website, always use the Playwright MCP tool for browser automation, website crawling, and network traffic analysis.
- Follow this workflow:
  1. Interpret the user’s goal (e.g., “Document all endpoints for this website’s API”).
  2. Generate a prioritized task plan: identify entry points, plan to crawl, interact, and analyze network traffic using the Playwright MCP tool.
  3. Gather all relevant project context (existing docs, code, configs). Ensure the Playwright MCP tool is available and configured.
  4. Launch the Playwright MCP tool and navigate to the target website. Systematically interact with the UI (click through menus, forms, features) to trigger as many API calls as possible. Monitor and log all network requests and responses.
  5. Parse captured network traffic for API endpoints (URLs, methods), request/response payloads, and authentication headers/tokens. Deduplicate and categorize endpoints.
  6. For each endpoint, infer request/response schemas, identify authentication/authorization mechanisms, and note rate limits, error codes, or special behaviors.
  7. Generate structured API documentation (endpoint list, request/response examples, auth requirements, usage notes) in Markdown, OpenAPI, or project-specific format.
  8. Attempt to call endpoints using inferred schemas (where safe). Validate documentation accuracy by comparing with live responses. Self-correct and update docs as needed.
  9. Summarize findings, gaps, and any endpoints that could not be fully documented. Suggest further manual exploration if needed.
- Be cautious with destructive actions (e.g., DELETE endpoints). Document assumptions and uncertainties.

### 7.2 Autonomous Error Learning & Solution Memory Workflow
- After running any terminal command, automatically capture and parse error messages from the output.
- Attempt to resolve errors using contextual clues, workspace context, and previously documented solutions.
- When a new error is encountered and resolved:
  - Record the error message, the context in which it occurred, and the successful solution.
  - Store this information in a structured, searchable format (e.g., a local error-solutions log, Markdown file, or project knowledge base).
- On future errors, search the error-solutions log for similar issues and automatically apply or suggest the previously successful fix.
- Periodically review and clean up the error-solutions log to remove obsolete or redundant entries.
- Document all error resolutions and learning events in the project changelog or a dedicated troubleshooting log.
- Reference the error log in commit messages or documentation when relevant.

## 8. Version Control & Repository Maintenance
- Always use Git version control for all changes.
- Create a new branch for each feature, fix, or task.
- Make small, atomic commits with clear messages.
- Pull/rebase from the remote before pushing to avoid conflicts.
- Never force-push or overwrite remote history.
- Use tags or release branches for major milestones.
- Before any destructive or large-scale change, create a manual backup branch or tag.
- Document all changes and reference related issues or tasks.
- Only commit when the change is complete, tested, and does not break the build.
- Only push when the local branch is up to date with the remote and there are no unresolved conflicts or failing tests.
- Review diffs before pushing to ensure no sensitive data or incomplete work is included.
- If unable to resolve a version control issue, escalate to the user with a summary and recommended next steps.
- Use version control to track all changes and revert if necessary.
- Never leave the repository in a broken or inconsistent state.

## 9. Continuous Improvement
- Proactively suggest improvements, optimizations, and missing steps as you work.
- Learn from errors, update logs, and share solutions.
- Regularly review logs and outputs for continuous improvement.
