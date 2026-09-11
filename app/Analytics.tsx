import BaiduTongjiTracker from "./BaiduTongjiTracker";

/**
 * 访问统计。构建时通过环境变量选择服务商，都不配置则什么都不渲染，
 * 本地开发或未配置时对站点零影响。
 *
 * - 百度统计（优先）：NEXT_PUBLIC_BAIDU_TONGJI_ID = hm.js 的站点 ID
 * - Umami（备选）：NEXT_PUBLIC_UMAMI_WEBSITE_ID，
 *   自建时可用 NEXT_PUBLIC_UMAMI_SRC 指定脚本地址
 */
export default function Analytics() {
  // 默认站点 ID；如需更换可用环境变量覆盖（ID 会出现在页面源码中，非敏感信息）
  const baiduId =
    process.env.NEXT_PUBLIC_BAIDU_TONGJI_ID ||
    "4ab702825934d35024963c1cce10100d";
  if (baiduId) {
    return (
      <>
        <script
          dangerouslySetInnerHTML={{
            __html: `var _hmt = _hmt || [];
(function() {
  var hm = document.createElement("script");
  hm.src = "https://hm.baidu.com/hm.js?${baiduId}";
  var s = document.getElementsByTagName("script")[0];
  s.parentNode.insertBefore(hm, s);
})();`,
          }}
        />
        <BaiduTongjiTracker />
      </>
    );
  }

  const umamiId = process.env.NEXT_PUBLIC_UMAMI_WEBSITE_ID;
  if (umamiId) {
    const src =
      process.env.NEXT_PUBLIC_UMAMI_SRC || "https://cloud.umami.is/script.js";
    return <script defer src={src} data-website-id={umamiId} />;
  }

  return null;
}
