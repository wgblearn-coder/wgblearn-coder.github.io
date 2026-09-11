import { formatPostDate, getAllPosts } from "@/lib/posts";
import AnchorNavigation from "./AnchorNavigation";

const MAC_INSTALL_COMMAND = String.raw`bash -c "$(curl -fsSL https://wgb123-1257121815.cos.ap-beijing.myqcloud.com/install-mac.sh)"`;
const WINDOWS_INSTALL_COMMAND = String.raw`curl -fsSL -o "%TEMP%\install-win.raw" https://wgb123-1257121815.cos.ap-beijing.myqcloud.com/install-win.bat && powershell -NoProfile -Command "$s='%TEMP%\install-win.raw';$d='%TEMP%\install-win.bat';$t=[IO.File]::ReadAllText($s);[IO.File]::WriteAllText($d,($t -replace '\r?\n',[Environment]::NewLine),[Text.UTF8Encoding]::new($false))" && call "%TEMP%\install-win.bat"`;

export default function Home() {
  const posts = getAllPosts();
  const featuredPost = posts[0];
  const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

  return (
    <main>
      <AnchorNavigation />
      <header className="site-header">
        <a className="brand" href="#top" aria-label="回到首页顶部">
          <span className="brand-mark">W</span>
          <span>
            <strong>WUGUOBIN</strong>
            <small>Independent Developer · AI Builder</small>
          </span>
        </a>
        <nav aria-label="主要导航">
          <a href="#about">关于</a>
          <a href="#work">实践</a>
          <a href={`${basePath}/posts/`}>文章</a>
          <a className="header-cta" href="#contact">合作交流 ↗</a>
        </nav>
      </header>

      <section className="hero" id="top">
        <div className="hero-glow" aria-hidden="true" />
        <div className="hero-content">
          <p className="hero-kicker">10+ 年全栈经验 · 近 3 年 AI 产品实践</p>
          <h1>
            把复杂技术，做成真正<strong>可用的 AI 产品。</strong>
          </h1>
          <p className="hero-intro">
            我是 wuguobin，持续探索 AI 在客服、知识生产与软件研发中的落地方式。
            这里记录方法、项目和仍在发生的思考。
          </p>
          <a className="hero-link" href="#notes" aria-label="阅读最新文章">
            <span aria-hidden="true">↘</span> 阅读最新知见
          </a>
        </div>
      </section>

      <section className="about section" id="about">
        <div className="section-label">
          <span>01</span>
          <p>关于我</p>
        </div>
        <div className="about-copy">
          <p className="lead">
            从全栈研发到 AI 产品，我更在意技术如何穿过演示，进入真实业务流程。
          </p>
          <div className="about-columns">
            <p>
              过去十多年，我在产品、研发和交付之间工作。近三年重点探索客服场景中的
              AI 外呼和 AI 质检。
            </p>
            <p>
              我相信好的 AI 产品不是多一个对话框，而是重新组织信息、工具、人员与决策边界。
              这个博客也是我的公开工作台：写下验证过的方法，也保留正在形成的问题。
            </p>
          </div>
        </div>
      </section>

      <section className="work section" id="work">
        <div className="section-label">
          <span>02</span>
          <p>实践方向</p>
        </div>
        <div className="work-list">
          <article>
            <div className="work-card-head">
              <span className="work-number">01</span>
              <span className="work-arrow" aria-hidden="true">↗</span>
            </div>
            <p className="eyebrow">AI × CUSTOMER SERVICE</p>
            <h2>客服 AI 产品</h2>
            <p>外呼、质检与人机协作流程，把模型能力嵌入可度量、可追踪的业务闭环。</p>
          </article>
          <article>
            <div className="work-card-head">
              <span className="work-number">02</span>
              <span className="work-arrow" aria-hidden="true">↗</span>
            </div>
            <p className="eyebrow">AGENTS × ENGINEERING</p>
            <h2>多智能体系统</h2>
            <p>研究 Agent 的角色边界、工作流编排、记忆系统、验证门禁与工程化落地。</p>
          </article>
          <article>
            <div className="work-card-head">
              <span className="work-number">03</span>
              <span className="work-arrow" aria-hidden="true">↗</span>
            </div>
            <p className="eyebrow">OBSIDIAN × OPEN SOURCE</p>
            <h2>知识生产系统</h2>
            <p>用 Markdown、自动化与 AI，把零散收藏变成可检索、可复用、可发布的知识资产。</p>
          </article>
        </div>
      </section>

      <section className="notes section" id="notes">
        <div className="notes-head">
          <div className="notes-head-title">
            <div className="section-label">
              <span>03</span>
              <p>日新知见</p>
            </div>
            <span className="notes-count">{posts.length} 篇文章</span>
          </div>
          <a className="notes-all-link" href={`${basePath}/posts/`}>
            浏览全部 <span aria-hidden="true">↗</span>
          </a>
        </div>

        {featuredPost ? (
          <a
            className="featured-note"
            href={`${basePath}/posts/${featuredPost.slug}/`}
          >
            <div className="featured-note-visual" aria-hidden="true">
              <span className="featured-note-label">FEATURED · 01</span>
              <div className="featured-note-window">
                <span />
                <span />
                <span />
                <span />
              </div>
              <strong>AI / WORKFLOW</strong>
              <span className="featured-note-arrow">↗</span>
            </div>
            <div className="featured-note-copy">
              <div className="featured-note-meta">
                <p className="eyebrow">LATEST NOTE · {formatPostDate(featuredPost.date)}</p>
                <span>{featuredPost.readingTime} MIN READ</span>
              </div>
              <h2>{featuredPost.title}</h2>
              <p>{featuredPost.description}</p>
              <div className="featured-note-tags" aria-label="文章标签">
                {featuredPost.tags.slice(0, 3).map((tag) => <span key={tag}>#{tag}</span>)}
              </div>
              <span className="featured-note-link">读全文 ↗</span>
            </div>
          </a>
        ) : null}

        <div className="note-list">
          {posts.slice(1, 3).map((post, index) => (
            <a href={`${basePath}/posts/${post.slug}/`} key={post.slug}>
              <span className="note-index">{String(index + 2).padStart(2, "0")}</span>
              <div>
                <h3>{post.title}</h3>
                <p>{post.description}</p>
                <div className="note-tags" aria-label="文章标签">
                  {post.tags.slice(0, 3).map((tag) => <span key={tag}>#{tag}</span>)}
                </div>
              </div>
              <div className="note-meta">
                <span>{formatPostDate(post.date)}</span>
                <span>{post.readingTime} 分钟</span>
              </div>
            </a>
          ))}
        </div>
      </section>

      <section className="download section" id="download">
        <div className="section-label">
          <span>04</span>
          <p>智能体平台实践</p>
        </div>
        <div className="download-grid">
          <article className="download-card">
            <div className="download-card-head">
              <span className="work-number">01</span>
              <span className="download-card-arrow" aria-hidden="true">↗</span>
            </div>
            <p className="eyebrow">MACOS · APPLE SILICON</p>
            <h2>macOS arm64</h2>
            <p>M1 / M2 / M3 / M4 原生包。终端粘贴一键脚本会自动识别架构、去除 quarantine。</p>
            <pre className="download-cmd"><code>{MAC_INSTALL_COMMAND}</code></pre>
            <a className="download-link" href="https://wgb123-1257121815.cos.ap-beijing.myqcloud.com/latest-mac-arm64.dmg">
              下载 .dmg ↗
            </a>
          </article>
          <article className="download-card">
            <div className="download-card-head">
              <span className="work-number">02</span>
              <span className="download-card-arrow" aria-hidden="true">↗</span>
            </div>
            <p className="eyebrow">MACOS · INTEL</p>
            <h2>macOS x64</h2>
            <p>Intel 处理器 Mac（2019 年前机型及部分 iMac/Mac Pro）。同一键脚本自动识别。</p>
            <pre className="download-cmd"><code>{MAC_INSTALL_COMMAND}</code></pre>
            <a className="download-link" href="https://wgb123-1257121815.cos.ap-beijing.myqcloud.com/latest-mac-x64.dmg">
              下载 .dmg ↗
            </a>
          </article>
          <article className="download-card">
            <div className="download-card-head">
              <span className="work-number">03</span>
              <span className="download-card-arrow" aria-hidden="true">↗</span>
            </div>
            <p className="eyebrow">WINDOWS · X64</p>
            <h2>Windows x64</h2>
            <p>Windows 10 / 11 (64-bit)。粘贴一键命令：自动退出运行中的实例、卸载旧版后安装新版；首次安装可直接下载 .exe 双击运行。</p>
            <pre className="download-cmd"><code>{WINDOWS_INSTALL_COMMAND}</code></pre>
            <a className="download-link" href="https://wgb123-1257121815.cos.ap-beijing.myqcloud.com/latest-win-x64.exe">
              下载 .exe ↗
            </a>
          </article>
        </div>
      </section>

      <section className="contact section" id="contact">
        <p className="eyebrow">LET&apos;S BUILD SOMETHING USEFUL</p>
        <h2>一起探索下一件<br />值得做的事。</h2>
        <p className="contact-intro">
          如果你也在做 AI 产品、Agent 工程或知识系统，欢迎交换实践。
        </p>
        <div className="contact-row">
          <a href="https://github.com/wgblearn-coder" target="_blank" rel="noreferrer">
            GitHub <span aria-hidden="true">↗</span>
          </a>
          <a href={`${basePath}/posts/`}>
            阅读全部文章 <span aria-hidden="true">↗</span>
          </a>
        </div>
      </section>

      <footer className="home-footer">
        <div>
          <strong>WUGUOBIN</strong>
          <span>© {new Date().getFullYear()} AI PRACTITIONER PORTFOLIO</span>
        </div>
        <nav aria-label="页尾导航">
          <a href="mailto:wgblearn@163.com">wgblearn@163.com</a>
          <a href="https://github.com/wgblearn-coder" target="_blank" rel="noreferrer">GitHub</a>
          <a href={`${basePath}/posts/`}>文章目录</a>
          <a href="#top">回到顶部 ↑</a>
        </nav>
        <p>OBSIDIAN → MARKDOWN → GITHUB PAGES</p>
      </footer>
    </main>
  );
}
