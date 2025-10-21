#!/bin/bash

# Phase 5 Frontend Bootstrap Script
# Automated setup for React + TypeScript + Testing Infrastructure

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project configuration
PROJECT_NAME="frontend"
BACKEND_URL="http://localhost:8000"
FRONTEND_PORT="5173"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  AI-CoScientist Phase 5 Frontend Bootstrap                 ║${NC}"
echo -e "${BLUE}║  React + TypeScript + TDD Infrastructure                   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to print status messages
print_status() {
    echo -e "${BLUE}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if Node.js is installed
print_status "Checking prerequisites..."
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    print_error "Node.js version must be 18 or higher. Current: $(node -v)"
    exit 1
fi

print_success "Node.js $(node -v) detected"

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    print_error "npm is not installed"
    exit 1
fi

print_success "npm $(npm -v) detected"

# Create frontend directory if it doesn't exist
print_status "Creating frontend directory..."
if [ -d "$PROJECT_NAME" ]; then
    print_warning "Frontend directory already exists"
    read -p "Do you want to delete and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$PROJECT_NAME"
        print_success "Removed existing frontend directory"
    else
        print_error "Aborted by user"
        exit 1
    fi
fi

# Create Vite project
print_status "Creating Vite + React + TypeScript project..."
npm create vite@latest "$PROJECT_NAME" -- --template react-ts
print_success "Vite project created"

cd "$PROJECT_NAME"

# Install core dependencies
print_status "Installing core dependencies..."
npm install

# Install UI libraries
print_status "Installing UI libraries (shadcn/ui dependencies)..."
npm install \
    tailwindcss \
    postcss \
    autoprefixer \
    class-variance-authority \
    clsx \
    tailwind-merge \
    lucide-react \
    @radix-ui/react-slot \
    @radix-ui/react-dialog \
    @radix-ui/react-dropdown-menu \
    @radix-ui/react-label \
    @radix-ui/react-progress \
    @radix-ui/react-select \
    @radix-ui/react-toast

print_success "UI libraries installed"

# Install state management and data fetching
print_status "Installing state management and data fetching..."
npm install \
    zustand \
    @tanstack/react-query \
    @tanstack/react-query-devtools \
    axios

print_success "State management libraries installed"

# Install routing
print_status "Installing React Router..."
npm install react-router-dom

print_success "React Router installed"

# Install file upload libraries
print_status "Installing file upload libraries..."
npm install react-dropzone

print_success "File upload libraries installed"

# Install diff viewing (use React 18 compatible version)
print_status "Installing diff viewing libraries..."
npm install diff prismjs

print_success "Diff libraries installed"

# Install WebSocket
print_status "Installing WebSocket libraries..."
npm install websocket

print_success "WebSocket libraries installed"

# Install testing dependencies
print_status "Installing testing infrastructure..."
npm install -D \
    vitest \
    @vitest/ui \
    @vitest/coverage-v8 \
    @testing-library/react \
    @testing-library/jest-dom \
    @testing-library/user-event \
    jsdom \
    msw \
    @playwright/test

print_success "Testing libraries installed"

# Install development tools
print_status "Installing development tools..."
npm install -D \
    eslint \
    @typescript-eslint/eslint-plugin \
    @typescript-eslint/parser \
    eslint-plugin-react-hooks \
    eslint-plugin-react-refresh \
    prettier \
    eslint-config-prettier \
    eslint-plugin-prettier \
    husky \
    lint-staged

print_success "Development tools installed"

# Initialize Tailwind CSS
print_status "Configuring Tailwind CSS..."
npx tailwindcss init -p

# Create Tailwind config
cat > tailwind.config.js << 'EOF'
/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    './pages/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './app/**/*.{ts,tsx}',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        "accordion-down": {
          from: { height: 0 },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: 0 },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
    },
  },
  plugins: [],
}
EOF

print_success "Tailwind configured"

# Update main CSS file
print_status "Setting up global styles..."
cat > src/index.css << 'EOF'
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --background: 0 0% 100%;
    --foreground: 222.2 84% 4.9%;

    --card: 0 0% 100%;
    --card-foreground: 222.2 84% 4.9%;

    --popover: 0 0% 100%;
    --popover-foreground: 222.2 84% 4.9%;

    --primary: 222.2 47.4% 11.2%;
    --primary-foreground: 210 40% 98%;

    --secondary: 210 40% 96.1%;
    --secondary-foreground: 222.2 47.4% 11.2%;

    --muted: 210 40% 96.1%;
    --muted-foreground: 215.4 16.3% 46.9%;

    --accent: 210 40% 96.1%;
    --accent-foreground: 222.2 47.4% 11.2%;

    --destructive: 0 84.2% 60.2%;
    --destructive-foreground: 210 40% 98%;

    --border: 214.3 31.8% 91.4%;
    --input: 214.3 31.8% 91.4%;
    --ring: 222.2 84% 4.9%;

    --radius: 0.5rem;
  }

  .dark {
    --background: 222.2 84% 4.9%;
    --foreground: 210 40% 98%;

    --card: 222.2 84% 4.9%;
    --card-foreground: 210 40% 98%;

    --popover: 222.2 84% 4.9%;
    --popover-foreground: 210 40% 98%;

    --primary: 210 40% 98%;
    --primary-foreground: 222.2 47.4% 11.2%;

    --secondary: 217.2 32.6% 17.5%;
    --secondary-foreground: 210 40% 98%;

    --muted: 217.2 32.6% 17.5%;
    --muted-foreground: 215 20.2% 65.1%;

    --accent: 217.2 32.6% 17.5%;
    --accent-foreground: 210 40% 98%;

    --destructive: 0 62.8% 30.6%;
    --destructive-foreground: 210 40% 98%;

    --border: 217.2 32.6% 17.5%;
    --input: 217.2 32.6% 17.5%;
    --ring: 212.7 26.8% 83.9%;
  }
}

@layer base {
  * {
    @apply border-border;
  }
  body {
    @apply bg-background text-foreground;
  }
}
EOF

print_success "Global styles configured"

# Create Vitest config
print_status "Configuring Vitest..."
cat > vitest.config.ts << 'EOF'
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/test/**',
        '**/*.test.{ts,tsx}',
        '**/*.spec.{ts,tsx}',
        '**/types.ts',
        '**/*.d.ts',
      ],
      thresholds: {
        lines: 80,
        functions: 80,
        branches: 80,
        statements: 80,
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
})
EOF

print_success "Vitest configured"

# Create Playwright config
print_status "Configuring Playwright..."
cat > playwright.config.ts << 'EOF'
import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
  ],
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:5173',
    reuseExistingServer: !process.env.CI,
  },
})
EOF

print_success "Playwright configured"

# Update tsconfig.json with path aliases
print_status "Configuring TypeScript paths..."
cat > tsconfig.json << 'EOF'
{
  "files": [],
  "references": [
    { "path": "./tsconfig.app.json" },
    { "path": "./tsconfig.node.json" }
  ],
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  }
}
EOF

cat > tsconfig.app.json << 'EOF'
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,

    /* Bundler mode */
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,
    "jsx": "react-jsx",

    /* Linting */
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,

    /* Path aliases */
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src"]
}
EOF

print_success "TypeScript configured"

# Update vite.config.ts with path aliases
print_status "Updating Vite config..."
cat > vite.config.ts << 'EOF'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
EOF

print_success "Vite configured"

# Create directory structure
print_status "Creating directory structure..."
mkdir -p src/{components,pages,hooks,api,lib,types,test,stores}
mkdir -p src/components/ui
mkdir -p e2e
mkdir -p public

print_success "Directory structure created"

# Create test setup file
print_status "Creating test setup..."
cat > src/test/setup.ts << 'EOF'
import '@testing-library/jest-dom'
import { afterEach, vi } from 'vitest'
import { cleanup } from '@testing-library/react'

// Cleanup after each test
afterEach(() => {
  cleanup()
})

// Mock window.matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
})
EOF

cat > src/test/utils.tsx << 'EOF'
import { ReactElement } from 'react'
import { render, RenderOptions } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
    },
  },
})

interface AllTheProvidersProps {
  children: React.ReactNode
}

function AllTheProviders({ children }: AllTheProvidersProps) {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>{children}</BrowserRouter>
    </QueryClientProvider>
  )
}

function customRender(
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>,
) {
  return render(ui, { wrapper: AllTheProviders, ...options })
}

export * from '@testing-library/react'
export { customRender as render }
EOF

print_success "Test setup created"

# Create MSW setup
print_status "Setting up MSW (Mock Service Worker)..."
mkdir -p src/mocks

cat > src/mocks/handlers.ts << 'EOF'
import { rest } from 'msw'

const API_BASE = 'http://localhost:8000'

export const handlers = [
  // Health check
  rest.get(`${API_BASE}/api/v1/health`, (req, res, ctx) => {
    return res(ctx.json({ status: 'ok' }))
  }),

  // Papers endpoints
  rest.get(`${API_BASE}/api/v1/papers`, (req, res, ctx) => {
    return res(ctx.json([]))
  }),

  rest.post(`${API_BASE}/api/v1/papers/upload`, (req, res, ctx) => {
    return res(
      ctx.status(201),
      ctx.json({
        paper_id: 'mock-paper-id',
        status: 'uploaded',
      })
    )
  }),

  rest.get(`${API_BASE}/api/v1/papers/:id`, (req, res, ctx) => {
    const { id } = req.params
    return res(
      ctx.json({
        id,
        title: 'Test Paper',
        status: 'uploaded',
      })
    )
  }),
]
EOF

cat > src/mocks/server.ts << 'EOF'
import { setupServer } from 'msw/node'
import { handlers } from './handlers'

export const server = setupServer(...handlers)
EOF

cat > src/mocks/browser.ts << 'EOF'
import { setupWorker } from 'msw/browser'
import { handlers } from './handlers'

export const worker = setupWorker(...handlers)
EOF

print_success "MSW configured"

# Create API client
print_status "Creating API client..."
cat > src/api/client.ts << 'EOF'
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized
      localStorage.removeItem('auth_token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export default apiClient
EOF

cat > src/api/papers.ts << 'EOF'
import { apiClient } from './client'

export interface Paper {
  id: string
  title: string
  status: 'uploaded' | 'analyzing' | 'complete'
  overall_score?: number
}

export const papersApi = {
  upload: async (file: File) => {
    const formData = new FormData()
    formData.append('file', file)

    const response = await apiClient.post('/api/v1/papers/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  getById: async (id: string): Promise<Paper> => {
    const response = await apiClient.get(`/api/v1/papers/${id}`)
    return response.data
  },

  list: async (): Promise<Paper[]> => {
    const response = await apiClient.get('/api/v1/papers')
    return response.data
  },
}
EOF

print_success "API client created"

# Create utility functions
print_status "Creating utility functions..."
cat > src/lib/utils.ts << 'EOF'
import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatScore(score: number): string {
  return score.toFixed(2)
}

export function getScoreColor(score: number): string {
  if (score >= 8.0) return 'text-green-600'
  if (score >= 7.0) return 'text-yellow-600'
  return 'text-red-600'
}
EOF

print_success "Utility functions created"

# Create basic components
print_status "Creating UI components..."
cat > src/components/ui/button.tsx << 'EOF'
import * as React from 'react'
import { Slot } from '@radix-ui/react-slot'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const buttonVariants = cva(
  'inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      variant: {
        default: 'bg-primary text-primary-foreground hover:bg-primary/90',
        destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/90',
        outline: 'border border-input bg-background hover:bg-accent hover:text-accent-foreground',
        secondary: 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
        ghost: 'hover:bg-accent hover:text-accent-foreground',
        link: 'text-primary underline-offset-4 hover:underline',
      },
      size: {
        default: 'h-10 px-4 py-2',
        sm: 'h-9 rounded-md px-3',
        lg: 'h-11 rounded-md px-8',
        icon: 'h-10 w-10',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : 'button'
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = 'Button'

export { Button, buttonVariants }
EOF

print_success "Button component created"

# Create example page
print_status "Creating example pages..."
cat > src/pages/Dashboard.tsx << 'EOF'
export function Dashboard() {
  return (
    <div className="container mx-auto p-8">
      <h1 className="text-4xl font-bold mb-4">
        AI-CoScientist Dashboard
      </h1>
      <p className="text-lg text-muted-foreground">
        Phase 5 Web UI - Ready for development
      </p>
    </div>
  )
}
EOF

cat > src/pages/NotFound.tsx << 'EOF'
import { Link } from 'react-router-dom'
import { Button } from '@/components/ui/button'

export function NotFound() {
  return (
    <div className="container mx-auto p-8 text-center">
      <h1 className="text-6xl font-bold mb-4">404</h1>
      <p className="text-xl mb-8">Page not found</p>
      <Button asChild>
        <Link to="/">Go Home</Link>
      </Button>
    </div>
  )
}
EOF

print_success "Example pages created"

# Update App.tsx
print_status "Creating router..."
cat > src/App.tsx << 'EOF'
import { Routes, Route } from 'react-router-dom'
import { Dashboard } from './pages/Dashboard'
import { NotFound } from './pages/NotFound'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Dashboard />} />
      <Route path="*" element={<NotFound />} />
    </Routes>
  )
}

export default App
EOF

# Update main.tsx
cat > src/main.tsx << 'EOF'
import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import App from './App.tsx'
import './index.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: 1,
    },
  },
})

// Enable MSW in development
if (import.meta.env.DEV) {
  import('./mocks/browser').then(({ worker }) => {
    worker.start({
      onUnhandledRequest: 'bypass',
    })
  })
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  </React.StrictMode>,
)
EOF

print_success "Router configured"

# Create example test
print_status "Creating example test..."
mkdir -p src/pages/__tests__

cat > src/pages/__tests__/Dashboard.test.tsx << 'EOF'
import { describe, it, expect } from 'vitest'
import { render, screen } from '@/test/utils'
import { Dashboard } from '../Dashboard'

describe('Dashboard', () => {
  it('should render dashboard title', () => {
    render(<Dashboard />)
    expect(screen.getByText(/AI-CoScientist Dashboard/i)).toBeInTheDocument()
  })

  it('should render subtitle', () => {
    render(<Dashboard />)
    expect(screen.getByText(/Phase 5 Web UI/i)).toBeInTheDocument()
  })
})
EOF

print_success "Example test created"

# Create E2E test example
print_status "Creating E2E test example..."
cat > e2e/example.spec.ts << 'EOF'
import { test, expect } from '@playwright/test'

test.describe('Basic Navigation', () => {
  test('should load homepage', async ({ page }) => {
    await page.goto('/')
    await expect(page.locator('h1')).toContainText('AI-CoScientist Dashboard')
  })

  test('should show 404 on invalid route', async ({ page }) => {
    await page.goto('/nonexistent')
    await expect(page.locator('h1')).toContainText('404')
  })
})
EOF

print_success "E2E test example created"

# Create environment file
print_status "Creating environment file..."
cat > .env.example << 'EOF'
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
EOF

cp .env.example .env

print_success "Environment file created"

# Update package.json scripts
print_status "Updating package.json scripts..."
npm pkg set scripts.dev="vite"
npm pkg set scripts.build="tsc && vite build"
npm pkg set scripts.preview="vite preview"
npm pkg set scripts.test="vitest"
npm pkg set scripts.test:ui="vitest --ui"
npm pkg set scripts.test:ci="vitest run --coverage"
npm pkg set scripts.test:changed="vitest related"
npm pkg set scripts.test:e2e="playwright test"
npm pkg set scripts.test:e2e:ui="playwright test --ui"
npm pkg set scripts.lint="eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0"
npm pkg set scripts.lint:fix="eslint . --ext ts,tsx --fix"
npm pkg set scripts.format="prettier --write \"src/**/*.{ts,tsx,css}\""
npm pkg set scripts.format:check="prettier --check \"src/**/*.{ts,tsx,css}\""
npm pkg set scripts.type-check="tsc --noEmit"
npm pkg set scripts.prepare="husky install"

print_success "Package scripts configured"

# Initialize git hooks
print_status "Setting up git hooks..."
npx husky install 2>/dev/null || true

cat > .husky/pre-commit << 'EOF'
#!/usr/bin/env sh
. "$(dirname -- "$0")/_/husky.sh"

npm run lint
npm run type-check
npm run test:changed
EOF

chmod +x .husky/pre-commit 2>/dev/null || true

print_success "Git hooks configured"

# Create ESLint config
print_status "Configuring ESLint..."
cat > .eslintrc.cjs << 'EOF'
module.exports = {
  root: true,
  env: { browser: true, es2020: true },
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:react-hooks/recommended',
    'prettier',
  ],
  ignorePatterns: ['dist', '.eslintrc.cjs'],
  parser: '@typescript-eslint/parser',
  plugins: ['react-refresh'],
  rules: {
    'react-refresh/only-export-components': [
      'warn',
      { allowConstantExport: true },
    ],
    '@typescript-eslint/no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
  },
}
EOF

print_success "ESLint configured"

# Create Prettier config
cat > .prettierrc << 'EOF'
{
  "semi": false,
  "singleQuote": true,
  "trailingComma": "es5",
  "printWidth": 100,
  "tabWidth": 2
}
EOF

print_success "Prettier configured"

# Create README
print_status "Creating README..."
cat > README.md << 'EOF'
# AI-CoScientist Frontend (Phase 5)

React + TypeScript + TDD web interface for AI-CoScientist paper improvement system.

## Quick Start

```bash
# Install dependencies (already done by bootstrap script)
npm install

# Start development server
npm run dev

# Open browser at http://localhost:5173
```

## Available Scripts

### Development
- `npm run dev` - Start dev server with HMR
- `npm run build` - Build for production
- `npm run preview` - Preview production build

### Testing
- `npm test` - Run unit tests in watch mode
- `npm run test:ui` - Open Vitest UI
- `npm run test:ci` - Run tests with coverage
- `npm run test:e2e` - Run E2E tests with Playwright
- `npm run test:e2e:ui` - Open Playwright UI

### Code Quality
- `npm run lint` - Run ESLint
- `npm run lint:fix` - Fix ESLint errors
- `npm run format` - Format code with Prettier
- `npm run type-check` - Check TypeScript types

## Project Structure

```
frontend/
├── src/
│   ├── components/     # React components
│   │   └── ui/        # shadcn/ui components
│   ├── pages/         # Page components (routes)
│   ├── hooks/         # Custom React hooks
│   ├── api/           # API client and endpoints
│   ├── lib/           # Utility functions
│   ├── types/         # TypeScript types
│   ├── stores/        # Zustand stores
│   ├── test/          # Test utilities
│   └── mocks/         # MSW mock handlers
├── e2e/               # Playwright E2E tests
└── public/            # Static assets
```

## Tech Stack

- **Framework**: React 18 + TypeScript
- **Build**: Vite 5
- **UI**: Tailwind CSS + shadcn/ui (Radix)
- **State**: Zustand
- **Data Fetching**: TanStack Query (React Query)
- **Routing**: React Router
- **Testing**: Vitest + Testing Library + Playwright
- **API Mocking**: MSW (Mock Service Worker)

## Development Workflow

1. **TDD Approach**: Write tests first (RED), implement (GREEN), refactor
2. **Component Development**:
   - Create component in `src/components/`
   - Write tests in `__tests__/` subdirectory
   - Ensure tests pass before committing
3. **Pre-commit Hooks**: Automatic lint, type-check, and test execution

## Backend Integration

The frontend connects to the FastAPI backend at `http://localhost:8000` (configurable via `.env`).

Backend API must be running for full functionality. For development, MSW mocks are used automatically.

## Testing Strategy

- **Unit Tests**: Component logic, hooks, utilities
- **Integration Tests**: API interactions, routing
- **E2E Tests**: Complete user workflows
- **Coverage Target**: >80% for all categories

## Next Steps

1. Review `claudedocs/PHASE5_WEB_UI_WORKFLOW.md` for implementation plan
2. Start with Phase 5.2: File Upload Component
3. Follow TDD methodology for each feature
4. Run tests frequently: `npm test`

## Resources

- [Vite Documentation](https://vitejs.dev/)
- [React Documentation](https://react.dev/)
- [Vitest Documentation](https://vitest.dev/)
- [Playwright Documentation](https://playwright.dev/)
- [shadcn/ui](https://ui.shadcn.com/)
EOF

print_success "README created"

# Install Playwright browsers
print_status "Installing Playwright browsers (this may take a few minutes)..."
npx playwright install chromium

print_success "Playwright browsers installed"

# Run initial tests
print_status "Running initial tests..."
npm test -- --run || print_warning "Some tests may need manual review"

print_success "Initial tests completed"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✓ Frontend Bootstrap Complete!                            ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📁 Project Location:${NC} $(pwd)"
echo -e "${BLUE}🌐 Development URL:${NC} http://localhost:5173"
echo -e "${BLUE}🔗 Backend API:${NC} http://localhost:8000"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo -e "  1. ${BLUE}cd $PROJECT_NAME${NC}"
echo -e "  2. ${BLUE}npm run dev${NC}          # Start development server"
echo -e "  3. ${BLUE}npm test${NC}             # Run tests in watch mode"
echo -e "  4. ${BLUE}npm run test:e2e:ui${NC}  # Open Playwright test UI"
echo ""
echo -e "${YELLOW}Available Commands:${NC}"
echo -e "  ${BLUE}npm run dev${NC}             Development server"
echo -e "  ${BLUE}npm test${NC}                Run unit tests"
echo -e "  ${BLUE}npm run test:ui${NC}         Open Vitest UI"
echo -e "  ${BLUE}npm run test:e2e${NC}        Run E2E tests"
echo -e "  ${BLUE}npm run lint${NC}            Check code quality"
echo -e "  ${BLUE}npm run build${NC}           Build for production"
echo ""
echo -e "${YELLOW}Resources:${NC}"
echo -e "  📖 Implementation Guide: ${BLUE}../claudedocs/PHASE5_WEB_UI_WORKFLOW.md${NC}"
echo -e "  📘 Project README: ${BLUE}./README.md${NC}"
echo ""
echo -e "${GREEN}Happy coding! 🚀${NC}"
