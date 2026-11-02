import { promises as fs } from "node:fs";
import { describe, expect, it } from "vitest";

describe("DGX Spark model images", () => {
  it("keeps the Hy-MT2 image tag aligned with its required Transformers runtime", async () => {
    const [dockerfile, compose] = await Promise.all([
      fs.readFile(new URL("../../services/dgx-models/hy-mt2/Dockerfile", import.meta.url), "utf8"),
      fs.readFile(new URL("../../compose.dgx.yaml", import.meta.url), "utf8"),
    ]);
    const runtimeVersion = dockerfile.match(/"transformers==([^"]+)"/)?.[1];
    const imageVersion = compose.match(/image:\s+sparkfire\/hy-mt2-1\.8b-fp8:transformers-([^\s]+)/)?.[1];

    expect(runtimeVersion).toBe("5.6.0");
    expect(imageVersion).toBe(runtimeVersion);
  });

  it("keeps the Parakeet image tag aligned with the pinned parakeet.cpp release", async () => {
    const [dockerfile, compose] = await Promise.all([
      fs.readFile(new URL("../../services/dgx-models/parakeet/Dockerfile", import.meta.url), "utf8"),
      fs.readFile(new URL("../../compose.dgx.yaml", import.meta.url), "utf8"),
    ]);

    expect(dockerfile.match(/ARG PARAKEET_REF=(\S+)/)?.[1]).toBe("v0.5.0");
    expect(dockerfile).toContain("ARG PARAKEET_COMMIT=1bfbebfaaf493866f49597cd3b7901959d395c60");
    expect(compose.match(/image:\s+sparkfire\/parakeet:parakeet\.cpp-(\S+)/)?.[1]).toBe("v0.5.0");
  });
});

describe("web search service", () => {
  it("pins SearXNG by digest, keeps it local and serves the JSON results the app reads", async () => {
    const [compose, settings, script] = await Promise.all([
      fs.readFile(new URL("../../compose.search.yaml", import.meta.url), "utf8"),
      fs.readFile(new URL("../../services/searxng/settings.yml", import.meta.url), "utf8"),
      fs.readFile(new URL("../../scripts/start-dgx-spark.sh", import.meta.url), "utf8"),
    ]);
    expect(compose).toMatch(/image:\s+searxng\/searxng:[\w.-]+@sha256:[0-9a-f]{64}\n/);
    expect(compose).toContain('"127.0.0.1:8338:8080"');
    expect(compose).toContain("WEB_SEARCH_BASE_URL: http://searxng:8080");
    expect(settings).toMatch(/formats:\s*\n\s*- html\s*\n\s*- json/);
    expect(settings).toContain("limiter: false");
    // Part of the stack, so it always starts.
    expect(script).toContain('start_service searxng "Web search (SearXNG)"');
    expect(script).toContain("--file compose.search.yaml");
  });
});

describe("Ternary Bonsai 2 27B", () => {
  const read = (file: string) => fs.readFile(new URL(`../../${file}`, import.meta.url), "utf8");

  it("builds llama-server from the PrismML fork at the pinned commit its image tag names", async () => {
    const [dockerfile, compose] = await Promise.all([read("services/dgx-models/llama-server/Dockerfile"), read("compose.dgx.yaml")]);
    expect(dockerfile.match(/ARG LLAMA_CPP_REF=(\S+)/)?.[1]).toBe("prism");
    expect(dockerfile).toContain("ARG LLAMA_CPP_REPO=https://github.com/PrismML-Eng/llama.cpp.git");
    expect(dockerfile).toContain("ARG LLAMA_CPP_COMMIT=1a07bfa5f4144274c8f1c9963821dd9d9a51854b");
    expect(dockerfile).toContain("-DCMAKE_CUDA_ARCHITECTURES=121");
    expect(compose.match(/image:\s+sparkfire\/llama-server:(\S+)/)?.[1]).toBe("prism-1a07bfa");
  });

  it("serves the files the start script downloads as bonsai-2-27b on 8331, with tool calls and image input", async () => {
    const [compose, script] = await Promise.all([read("compose.dgx.yaml"), read("scripts/start-dgx-spark.sh")]);
    const service = compose.slice(compose.indexOf("  bonsai:\n"), compose.indexOf("  parakeet:\n"));
    for (const file of ["Ternary-Bonsai-2-27B-PTQ1_0.gguf", "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"]) {
      expect(service).toContain(`/models/prism-ml/Ternary-Bonsai-2-27B-gguf/${file}`);
      expect(script).toContain(`"${file}"`);
    }
    expect(script).toContain("94dd51cdbbec5309ed8b76a12f3268bc4fdceb93");
    expect(service).toMatch(/- --alias\n\s+- bonsai-2-27b\n/);
    expect(service).toContain("- --mmproj");
    expect(service).toContain("- --jinja");
    expect(service).toContain("- --image-max-tokens");
    expect(service).toMatch(/- --image-max-tokens\n\s+- "2048"\n/);
    expect(service).toMatch(/- --port\n\s+- "8331"\n/);
    expect(script).toContain('start_service bonsai "LLM (Ternary Bonsai 2 27B)" "http://127.0.0.1:8331/health"');
  });
});

describe("the fixed stack", () => {
  it("starts music with the stack and keeps no other model choices", async () => {
    const [compose, script] = await Promise.all([
      fs.readFile(new URL("../../compose.dgx.yaml", import.meta.url), "utf8"),
      fs.readFile(new URL("../../scripts/start-dgx-spark.sh", import.meta.url), "utf8"),
    ]);
    expect(compose).not.toContain("profiles: [music]");
    expect(compose).not.toContain("unlimited-ocr:");
    expect(compose).toContain("IMAGE_BACKEND: qwen-image-2.1-turbo");
    expect(script).toContain('start_service music-generation "Music generation (YuE2)"');
    // Video stays optional where FastH3's license does not apply.
    expect(script).toContain("--video-backend NAME");
    expect(script).not.toMatch(/--(ocr|image|music|search)-backend NAME/);
  });
});

describe("names", () => {
  it("names the compose project, every container and every image the stack builds sparkfire", async () => {
    const read = (file: string) => fs.readFile(new URL(`../../${file}`, import.meta.url), "utf8");
    const [composeFiles, dgxScript, workspaceScript] = await Promise.all([
      Promise.all(["compose.yaml", "compose.spark.yaml", "compose.dgx.yaml", "compose.search.yaml"].map(read)),
      read("scripts/start-dgx-spark.sh"),
      read("scripts/start-sparkfire.sh"),
    ]);
    const compose = composeFiles.join("\n");
    const containers = [...compose.matchAll(/container_name:\s+(\S+)/g)].map((match) => match[1]);
    expect(containers.length).toBeGreaterThanOrEqual(10);
    expect(containers.every((name) => name.startsWith("sparkfire-"))).toBe(true);
    // A service block that builds its image names it; pulled images (Redis, vLLM, SearXNG) keep their publishers'.
    const built = compose.split(/\n(?= {2}[\w-]+:\n)/).filter((block) => /\n {4}build:/.test(block)).flatMap((block) => block.match(/\n {4}image:\s+(\S+)/)?.[1] ?? []);
    expect(built.length).toBeGreaterThanOrEqual(9);
    expect(built.every((image) => image.startsWith("sparkfire/"))).toBe(true);
    for (const script of [dgxScript, workspaceScript]) expect(script).toContain("--project-name sparkfire");
  });
});
