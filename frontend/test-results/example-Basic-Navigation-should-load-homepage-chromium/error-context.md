# Page snapshot

```yaml
- generic [ref=e3]:
  - generic [ref=e4]: "[plugin:vite:css] [postcss] tailwindcss: /Users/jiookcha/Documents/git/AI-CoScientist/frontend/src/index.css:1:1: Cannot apply unknown utility class `bg-background`. Are you using CSS modules or similar and missing `@reference`? https://tailwindcss.com/docs/functions-and-directives#reference-directive"
  - generic [ref=e5]: /Users/jiookcha/Documents/git/AI-CoScientist/frontend/src/index.css:1:0
  - generic [ref=e6]: 1 | @tailwind base; | ^ 2 | @tailwind components; 3 | @tailwind utilities;
  - generic [ref=e7]: at Input.error (/Users/jiookcha/Documents/git/AI-CoScientist/frontend/node_modules/postcss/lib/input.js:135:16) at Root.error (/Users/jiookcha/Documents/git/AI-CoScientist/frontend/node_modules/postcss/lib/node.js:146:32) at Object.Once (/Users/jiookcha/Documents/git/AI-CoScientist/frontend/node_modules/@tailwindcss/postcss/dist/index.js:10:6913) at async LazyResult.runAsync (/Users/jiookcha/Documents/git/AI-CoScientist/frontend/node_modules/postcss/lib/lazy-result.js:293:11) at async runPostCSS (file:///Users/jiookcha/Documents/git/AI-CoScientist/frontend/node_modules/vite/dist/node/chunks/dep-Chhhsdoe.js:32166:19) at async compilePostCSS (file:///Users/jiookcha/Documents/git/AI-CoScientist/frontend/node_modules/vite/dist/node/chunks/dep-Chhhsdoe.js:32150:6) at async compileCSS (file:///Users/jiookcha/Documents/git/AI-CoScientist/frontend/node_modules/vite/dist/node/chunks/dep-Chhhsdoe.js:32080:26) at async TransformPluginContext.handler (file:///Users/jiookcha/Documents/git/AI-CoScientist/frontend/node_modules/vite/dist/node/chunks/dep-Chhhsdoe.js:31613:54) at async EnvironmentPluginContainer.transform (file:///Users/jiookcha/Documents/git/AI-CoScientist/frontend/node_modules/vite/dist/node/chunks/dep-Chhhsdoe.js:31018:14) at async loadAndTransform (file:///Users/jiookcha/Documents/git/AI-CoScientist/frontend/node_modules/vite/dist/node/chunks/dep-Chhhsdoe.js:26143:26)
  - generic [ref=e8]:
    - text: Click outside, press Esc key, or fix the code to dismiss.
    - text: You can also disable this overlay by setting
    - code [ref=e9]: server.hmr.overlay
    - text: to
    - code [ref=e10]: "false"
    - text: in
    - code [ref=e11]: vite.config.ts
    - text: .
```