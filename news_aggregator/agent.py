import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse
from typing import TYPE_CHECKING

import feedparser # type: ignore
import requests # type: ignore
from bs4 import BeautifulSoup # type: ignore
from dotenv import load_dotenv # type: ignore
from langchain_core.prompts import PromptTemplate # type: ignore
from langchain_openai import ChatOpenAI # type: ignore

load_dotenv(".env")

try:
    from langgraph.graph import StateGraph # type: ignore
    from typing import TypedDict
    
    class NewsAggregatorState(TypedDict):
        """State for the news aggregator graph"""
        topic: str
        rss_urls: list
        website_urls: list
        results: dict
    
    def create_news_aggregator_graph():
        """Create a LangGraph graph for news aggregation"""
        graph = StateGraph(NewsAggregatorState)
        
        def aggregate_news_node(state: NewsAggregatorState):
            """Main node that runs the news aggregator"""
            aggregator = NewsAggregator(topic=state["topic"])
            results = aggregator.run(
                rss_urls=state.get("rss_urls", []),
                website_urls=state.get("website_urls", [])
            )
            return {"results": results}
        
        graph.add_node("aggregate", aggregate_news_node)
        graph.set_entry_point("aggregate")
        graph.set_finish_point("aggregate")
        
        return graph.compile()
    
    app = create_news_aggregator_graph()
except ImportError:
    app = None


class Config:
    CUSTOM_BASE_URL = os.environ.get("CUSTOM_BASE_URL")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    OPENAI_MODEL = "gpt-4o-mini"
    OPENAI_TEMPERATURE = 0
    LANGSMITH_TRACING = os.environ.get("LANGSMITH_TRACING")
    LANGCHAIN_PROJECT = os.environ.get("LANGSMITH_PROJECT")
    LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", "")
    REQUEST_TIMEOUT = 10
    MAX_ARTICLES_PER_SOURCE = 5
    MAX_CONTENT_LENGTH = 1000
    OUTPUT_DIR = "output"
    
    # Website crawling config
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    MAX_CRAWL_DEPTH = 2  # Độ sâu crawl tối đa
    MAX_ARTICLES_PER_WEBSITE = 10
    
    @classmethod
    def setup_environment(cls):
        if cls.LANGSMITH_TRACING:
            os.environ["LANGSMITH_TRACING"] = cls.LANGSMITH_TRACING
        if cls.LANGCHAIN_PROJECT:
            os.environ["LANGCHAIN_PROJECT"] = cls.LANGCHAIN_PROJECT
        if cls.LANGCHAIN_API_KEY:
            os.environ["LANGCHAIN_API_KEY"] = cls.LANGCHAIN_API_KEY
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
    
    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError("❌ OPENAI_API_KEY chưa được thiết lập!")
        print("✅ Cấu hình hợp lệ")


class WebsiteCrawler:
    """Class chuyên crawl tin tức từ website"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': Config.USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
        })
        self.visited_urls = set()
    
    def is_valid_url(self, url: str) -> bool:
        """Kiểm tra URL hợp lệ"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def get_page_content(self, url: str) -> Optional[BeautifulSoup]:
        """Lấy nội dung trang web"""
        try:
            response = self.session.get(url, timeout=Config.REQUEST_TIMEOUT)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            print(f"  ⚠️ Lỗi tải trang: {str(e)}")
            return None
    
    def extract_article_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Trích xuất link bài viết từ trang"""
        links = []
        
        # Tìm các thẻ a có chứa link bài viết
        # Thường có class như: article, post, news-item, title, headline
        selectors = [
            'a[href*="/article"]',
            'a[href*="/post"]',
            'a[href*="/news"]',
            'a[href*="/tin-tuc"]',
            'a[href*="/bai-viet"]',
            'article a',
            '.article a',
            '.post a',
            '.news-item a',
            'h2 a',
            'h3 a'
        ]
        
        for selector in selectors:
            for link in soup.select(selector):
                href = link.get('href')
                if href:
                    full_url = urljoin(base_url, href)
                    if self.is_valid_url(full_url) and full_url not in self.visited_urls:
                        links.append(full_url)
        
        return list(set(links))[:Config.MAX_ARTICLES_PER_WEBSITE]
    
    def extract_article_content(self, soup: BeautifulSoup, url: str) -> Optional[Dict]:
        """Trích xuất nội dung bài viết"""
        try:
            # Tìm tiêu đề
            title = None
            for selector in ['h1', 'h2.title', '.article-title', '.post-title', 'h1.headline']:
                title_tag = soup.select_one(selector)
                if title_tag:
                    title = title_tag.get_text(strip=True)
                    break
            
            if not title:
                title = soup.title.string if soup.title else "Không có tiêu đề"
            
            # Tìm nội dung chính
            content = ""
            content_selectors = [
                'article',
                '.article-content',
                '.post-content',
                '.entry-content',
                '.news-content',
                '.content-detail',
                'div[itemprop="articleBody"]',
                '.detail-content'
            ]
            
            for selector in content_selectors:
                content_tag = soup.select_one(selector)
                if content_tag:
                    # Loại bỏ các thẻ không cần thiết
                    for tag in content_tag.find_all(['script', 'style', 'iframe', 'nav', 'aside']):
                        tag.decompose()
                    
                    # Lấy text từ các đoạn văn
                    paragraphs = content_tag.find_all(['p', 'div'])
                    content = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                    break
            
            # Nếu không tìm thấy content, lấy từ toàn bộ body
            if not content:
                body = soup.find('body')
                if body:
                    paragraphs = body.find_all('p')
                    content = ' '.join([p.get_text(strip=True) for p in paragraphs[:5]])
            
            # Tìm ngày đăng
            published = None
            date_selectors = [
                'time',
                '.published-date',
                '.post-date',
                '.date',
                'meta[property="article:published_time"]',
                'meta[name="pubdate"]'
            ]
            
            for selector in date_selectors:
                date_tag = soup.select_one(selector)
                if date_tag:
                    published = date_tag.get('datetime') or date_tag.get('content') or date_tag.get_text(strip=True)
                    break
            
            # Lấy domain làm source
            domain = urlparse(url).netloc
            
            if len(content) < 50:
                return None
            
            return {
                'title': title[:200],
                'link': url,
                'published': published or 'N/A',
                'summary': content[:1000],
                'source': domain,
                'source_type': 'website'
            }
            
        except Exception as e:
            print(f"  ⚠️ Lỗi trích xuất: {str(e)}")
            return None
    
    def crawl_website(self, url: str, topic: str = "") -> List[Dict]:
        """Crawl tin tức từ một website"""
        articles = []
        
        print(f"\n🌐 Đang crawl website: {url}")
        
        # Lấy trang chủ
        soup = self.get_page_content(url)
        if not soup:
            return articles
        
        # Trích xuất các link bài viết
        article_links = self.extract_article_links(soup, url)
        print(f"  📋 Tìm thấy {len(article_links)} link bài viết")
        
        # Crawl từng bài viết
        for idx, link in enumerate(article_links, 1):
            if link in self.visited_urls:
                continue
            
            self.visited_urls.add(link)
            print(f"  [{idx}/{len(article_links)}] Đang xử lý: {link[:70]}...")
            
            article_soup = self.get_page_content(link)
            if not article_soup:
                continue
            
            article = self.extract_article_content(article_soup, link)
            if article:
                articles.append(article)
                print(f"    ✅ Đã lấy: {article['title'][:60]}...")
        
        print(f"  ✅ Crawl xong: {len(articles)} bài viết")
        return articles


class NewsAggregator:
    """Hệ thống thu thập và tổng hợp tin tức tự động"""
    
    def __init__(self, topic: str):
        self.topic = topic
        self.llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=Config.OPENAI_TEMPERATURE,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.CUSTOM_BASE_URL,
        )
        self.crawler = WebsiteCrawler()
        self.articles = []
        print(f"🎯 Khởi tạo NewsAggregator cho chủ đề: '{topic}'")
    
    def test_rss_feed(self, url: str) -> bool:
        """Kiểm tra xem URL có phải RSS feed hợp lệ không"""
        try:
            print(f"🔍 Đang test RSS: {url}")
            feed = feedparser.parse(url)
            if feed.entries:
                print(f"  ✅ RSS hợp lệ! Tìm thấy {len(feed.entries)} bài viết")
                print(f"  📰 Tiêu đề feed: {feed.feed.get('title', 'N/A')}")
                if feed.entries:
                    print(f"  📝 Bài đầu tiên: {feed.entries[0].get('title', 'N/A')[:60]}...")
                return True
            else:
                print(f"  ❌ Không phải RSS feed hoặc không có bài viết")
                if feed.bozo:
                    print(f"  ⚠️ Lỗi parse: {feed.bozo_exception}")
                return False
        except Exception as e:
            print(f"  ❌ Lỗi: {str(e)}")
            return False
    
    def test_website(self, url: str) -> bool:
        """Kiểm tra xem website có thể crawl được không"""
        try:
            print(f"🔍 Đang test website: {url}")
            soup = self.crawler.get_page_content(url)
            if soup:
                links = self.crawler.extract_article_links(soup, url)
                print(f"  ✅ Website hợp lệ! Tìm thấy {len(links)} link bài viết")
                return len(links) > 0
            return False
        except Exception as e:
            print(f"  ❌ Lỗi: {str(e)}")
            return False
    
    def fetch_rss_feeds(self, rss_urls: List[str]) -> List[Dict]:
        """Thu thập tin tức từ các nguồn RSS"""
        articles = []
        
        if not rss_urls:
            return articles
        
        print(f"\n🔍 Đang thu thập tin tức từ {len(rss_urls)} nguồn RSS...")
        print("-" * 80)
        
        for idx, url in enumerate(rss_urls, 1):
            try:
                print(f"[{idx}/{len(rss_urls)}] Đang xử lý: {url}")
                feed = feedparser.parse(url)
                
                if not feed.entries:
                    print(f"  ⚠️ Không có bài viết nào")
                    continue
                
                source_name = feed.feed.get("title", url)
                count = 0
                
                for entry in feed.entries[:Config.MAX_ARTICLES_PER_SOURCE]:
                    article = {
                        'title': entry.get('title', 'Không có tiêu đề'),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', 'N/A'),
                        'summary': entry.get('summary', entry.get('description', '')),
                        'source': source_name,
                        'source_type': 'rss'
                    }
                    articles.append(article)
                    count += 1
                
                print(f"  ✅ Đã lấy {count} bài viết")
                
            except Exception as e:
                print(f"  ❌ Lỗi: {str(e)}")
        
        print(f"\n✅ Tổng số bài viết từ RSS: {len(articles)}")
        return articles
    
    def fetch_websites(self, website_urls: List[str]) -> List[Dict]:
        """Thu thập tin tức từ các website"""
        articles = []
        
        if not website_urls:
            return articles
        
        print(f"\n🌐 Đang crawl tin tức từ {len(website_urls)} website...")
        print("-" * 80)
        
        for idx, url in enumerate(website_urls, 1):
            try:
                print(f"[{idx}/{len(website_urls)}] Đang crawl: {url}")
                crawled_articles = self.crawler.crawl_website(url, self.topic)
                articles.extend(crawled_articles)
            except Exception as e:
                print(f"  ❌ Lỗi: {str(e)}")
        
        print(f"\n✅ Tổng số bài viết từ website: {len(articles)}")
        return articles
    
    def filter_by_topic(self, articles: List[Dict]) -> List[Dict]:
        """Lọc tin tức theo chủ đề sử dụng LangChain"""
        print(f"\n🔎 Đang lọc tin tức liên quan đến '{self.topic}'...")
        print("-" * 80)
        
        filter_prompt = PromptTemplate(
            input_variables=["topic", "title", "summary"],
            template="""Bạn là một chuyên gia phân loại tin tức.

            Phân tích xem bài viết sau có TRỰC TIẾP liên quan đến chủ đề "{topic}" không.

            Tiêu đề: {title}
            Tóm tắt: {summary}

            Chỉ trả lời "Có liên quan" nếu bài viết có liên quan TRỰC TIẾP và RÕ RÀNG đến chủ đề.
            Trả lời "Không liên quan" nếu chỉ liên quan gián tiếp hoặc đề cập qua loa.

            Câu trả lời (chỉ Có liên quan hoặc Không liên quan):"""
        )
        
        chain = filter_prompt | self.llm
        filtered_articles = []
        
        for idx, article in enumerate(articles, 1):
            try:
                result = chain.invoke({
                    "topic": self.topic,
                    "title": article['title'],
                    "summary": article['summary'][:500]
                })
                if "CÓ LIÊN QUAN" in result.content.upper():
                    filtered_articles.append(article)
                    source_icon = "📡" if article.get('source_type') == 'rss' else "🌐"
                    print(f"[{idx}/{len(articles)}] ✅ {source_icon} Giữ: {article['title'][:60]}...")
                else:
                    print(f"[{idx}/{len(articles)}] ❌ Loại: {article['title'][:60]}...")
                    
            except Exception as e:
                print(f"[{idx}/{len(articles)}] ⚠️ Lỗi: {str(e)}")
        
        print(f"\n✅ Số bài viết sau khi lọc: {len(filtered_articles)}/{len(articles)}")
        return filtered_articles
    
    def summarize_articles(self, articles: List[Dict]) -> str:
        """Tóm tắt và tổng hợp các tin tức"""
        print(f"\n📝 Đang tóm tắt {len(articles)} bài viết...")
        print("-" * 80)
        
        # Chuẩn bị nội dung
        articles_text = ""
        for i, article in enumerate(articles, 1):
            source_type = "📡 RSS" if article.get('source_type') == 'rss' else "🌐 Website"
            articles_text += f"""
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            BÀI VIẾT {i} ({source_type})
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            Tiêu đề: {article['title']}
            Nguồn: {article['source']}
            Ngày: {article.get('published', 'N/A')}
            Tóm tắt: {article['summary'][:400]}
            """
            
            summary_prompt = PromptTemplate(
                input_variables=["topic", "articles", "count"],
                template="""Bạn là một nhà phân tích tin tức chuyên nghiệp.

            Hãy tạo một BÁO CÁO TỔNG HỢP về chủ đề "{topic}" dựa trên {count} bài viết sau:

            {articles}

            YÊU CẦU BÁO CÁO:

            1. TỔNG QUAN
            - Nêu tình hình chung về chủ đề
            - Các sự kiện, xu hướng chính đang diễn ra

            2. PHÂN TÍCH CHI TIẾT
            - Phân tích các khía cạnh quan trọng
            - Dẫn chứng cụ thể từ các nguồn tin
            - So sánh quan điểm khác nhau (nếu có)

            3. XU HƯỚNG VÀ DỰ BÁO
            - Các xu hướng đáng chú ý
            - Tác động tiềm năng
            - Dự báo phát triển

            4. KẾT LUẬN
            - Tổng kết các điểm chính
            - Đánh giá tổng quan

            LƯU Ý:
            - Viết bằng tiếng Việt, chuyên nghiệp
            - Trích dẫn nguồn khi cần thiết
            - Khách quan, không thiên vị
            - Độ dài: 800-1200 từ

            BÁO CÁO:"""
        )
        
        chain = summary_prompt | self.llm
        summary = chain.invoke({
            "topic": self.topic,
            "articles": articles_text,
            "count": len(articles)
        }).content
        
        print("\n✅ Hoàn thành tóm tắt")
        return summary
    
    def evaluate_quality(self, summary: str, articles: List[Dict]) -> Dict:
        """Đánh giá chất lượng tóm tắt sử dụng LangSmith"""
        print("\n⭐ Đang đánh giá chất lượng báo cáo...")
        print("-" * 80)
        
        eval_prompt = PromptTemplate(
            input_variables=["summary", "num_articles", "word_count"],
            template="""Bạn là một chuyên gia đánh giá chất lượng báo cáo.

            Hãy đánh giá báo cáo tổng hợp tin tức sau:

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            BÁO CÁO CẦN ĐÁNH GIÁ
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            {summary}

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            THÔNG TIN
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            - Số bài viết nguồn: {num_articles}
            - Số từ: {word_count}

            ĐÁNH GIÁ theo thang điểm 1-10 cho các tiêu chí sau:

            1. ACCURACY (Độ chính xác): Thông tin có chính xác, không sai lệch?
            2. COMPLETENESS (Độ đầy đủ): Có bao quát các khía cạnh chính không?
            3. CLARITY (Độ rõ ràng): Dễ hiểu, mạch lạc, logic?
            4. OBJECTIVITY (Tính khách quan): Trung lập, không thiên vị?
            5. VALUE (Giá trị): Hữu ích, có insight hay?

            Trả lời ĐÚNG định dạng JSON sau (không thêm gì khác):
            {{
                "accuracy": 8,
                "completeness": 7,
                "clarity": 9,
                "objectivity": 8,
                "value": 7,
                "overall": 7.8,
                "feedback": "Báo cáo tốt, rõ ràng..."
            }}

            JSON:"""
        )
        
        chain = eval_prompt | self.llm
        
        try:
            word_count = len(summary.split())
            result = chain.invoke({
                "summary": summary[:2000],
                "num_articles": len(articles),
                "word_count": word_count
            }).content
            
            # Trích xuất JSON
            start_idx = result.find('{')
            end_idx = result.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("Không tìm thấy JSON trong kết quả")
            
            json_str = result[start_idx:end_idx]
            evaluation = json.loads(json_str)
            
            # In kết quả
            print(f"\n📊 KẾT QUẢ ĐÁNH GIÁ:")
            print(f"  • Độ chính xác: {evaluation['accuracy']}/10")
            print(f"  • Độ đầy đủ: {evaluation['completeness']}/10")
            print(f"  • Độ rõ ràng: {evaluation['clarity']}/10")
            print(f"  • Tính khách quan: {evaluation['objectivity']}/10")
            print(f"  • Giá trị: {evaluation['value']}/10")
            print(f"  • TỔNG THỂ: {evaluation['overall']}/10")
            print(f"\n💬 Nhận xét: {evaluation['feedback']}")
            
            return evaluation
            
        except Exception as e:
            print(f"⚠️ Không thể đánh giá: {str(e)}")
            return {
                "error": str(e),
                "overall": 0
            }
    
    @staticmethod
    def _auto_download_file(filepath: str, filename: str, mimetype: str):
        """
        Tự động tải file về máy (auto-download)
        Hỗ trợ cả Jupyter Notebook và terminal
        """
        import base64
        
        try:
            try:
                from IPython.display import HTML, display, Javascript  # type: ignore
                in_jupyter = True
            except ImportError:
                in_jupyter = False
            
            if in_jupyter:
                # Đọc file
                with open(filepath, 'rb') as f:
                    file_content = f.read()
                
                b64 = base64.b64encode(file_content).decode()
                
                # Tạo JavaScript để tự động tải file xuống
                js_download = f"""
                var link = document.createElement('a');
                link.href = 'data:{mimetype};base64,{b64}';
                link.download = '{filename}';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                """
                
                # Thực thi JavaScript để tải file
                display(Javascript(js_download))
                
                # Hiển thị thông báo và link backup
                download_link = f'''
                <div style="padding: 10px; background-color: #e8f5e9; border-left: 4px solid #4caf50; border-radius: 5px; margin: 10px 0;">
                    <p style="margin: 0; color: #2e7d32; font-weight: bold;">✅ File đang được tải xuống tự động...</p>
                    <p style="margin: 5px 0 0 0; font-size: 0.9em;">
                        Nếu không tự động tải, 
                        <a href="data:{mimetype};base64,{b64}" 
                        download="{filename}"
                        style="color: #1976d2; text-decoration: underline;">
                            click vào đây
                        </a>
                    </p>
                </div>
                '''
                display(HTML(download_link))
                print(f"📥 Đang tải xuống: {filename}")
                
            else:
                # Terminal: Mở file explorer/finder tại vị trí file
                abs_path = os.path.abspath(filepath)
                print(f"📁 File đã được lưu tại: {abs_path}")
                
                # Thử mở file explorer (tùy hệ điều hành)
                try:
                    import platform
                    import subprocess
                    
                    system = platform.system()
                    if system == 'Windows':
                        subprocess.Popen(['explorer', '/select,', abs_path])
                        print("📂 Đã mở File Explorer")
                    elif system == 'Darwin':  # macOS
                        subprocess.Popen(['open', '-R', abs_path])
                        print("📂 Đã mở Finder")
                    elif system == 'Linux':
                        # Thử mở file manager
                        subprocess.Popen(['xdg-open', os.path.dirname(abs_path)])
                        print("📂 Đã mở File Manager")
                except Exception as e:
                    print(f"⚠️ Không thể mở file explorer: {e}")
                
        except Exception as e:
            print(f"⚠️ Không thể tự động tải: {e}")
            print(f"📁 File đã được lưu tại: {os.path.abspath(filepath)}")

    def export_to_txt(self, summary: str, articles: List[Dict], evaluation: Dict, filename: str):
        """Xuất báo cáo ra file TXT và tự động tải về"""
        filepath = os.path.join(Config.OUTPUT_DIR, filename)
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Header
            f.write("╔" + "═" * 78 + "╗\n")
            f.write(f"║{'BÁO CÁO TỔNG HỢP TIN TỨC'.center(78)}║\n")
            f.write(f"║{self.topic.upper().center(78)}║\n")
            f.write("╚" + "═" * 78 + "╝\n\n")
            
            # Thông tin
            f.write(f"📅 Ngày tạo: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"📊 Số bài viết: {len(articles)}\n")
            
            # Thống kê nguồn
            rss_count = sum(1 for a in articles if a.get('source_type') == 'rss')
            web_count = sum(1 for a in articles if a.get('source_type') == 'website')
            f.write(f"📡 Từ RSS: {rss_count} bài\n")
            f.write(f"🌐 Từ Website: {web_count} bài\n")
            f.write(f"⭐ Điểm đánh giá: {evaluation.get('overall', 'N/A')}/10\n")
            f.write("\n" + "━" * 80 + "\n\n")
            
            # Tóm tắt
            f.write("📝 TÓM TẮT TỔNG HỢP\n")
            f.write("━" * 80 + "\n\n")
            f.write(summary)
            f.write("\n\n" + "━" * 80 + "\n\n")
            
            # Đánh giá
            if 'error' not in evaluation:
                f.write("⭐ ĐÁNH GIÁ CHẤT LƯỢNG\n")
                f.write("━" * 80 + "\n\n")
                f.write(f"• Độ chính xác: {evaluation['accuracy']}/10\n")
                f.write(f"• Độ đầy đủ: {evaluation['completeness']}/10\n")
                f.write(f"• Độ rõ ràng: {evaluation['clarity']}/10\n")
                f.write(f"• Tính khách quan: {evaluation['objectivity']}/10\n")
                f.write(f"• Giá trị: {evaluation['value']}/10\n")
                f.write(f"• TỔNG THỂ: {evaluation['overall']}/10\n\n")
                f.write(f"💬 Nhận xét: {evaluation['feedback']}\n")
                f.write("\n" + "━" * 80 + "\n\n")
            
            # Danh sách bài viết
            f.write("📚 DANH SÁCH BÀI VIẾT NGUỒN\n")
            f.write("━" * 80 + "\n\n")
            
            for i, article in enumerate(articles, 1):
                source_icon = "📡" if article.get('source_type') == 'rss' else "🌐"
                f.write(f"{i}. {article['title']}\n")
                f.write(f"   {source_icon} Nguồn: {article['source']} ({article.get('source_type', 'unknown').upper()})\n")
                f.write(f"   🔗 Link: {article['link']}\n")
                f.write(f"   📅 Ngày: {article.get('published', 'N/A')}\n")
                summary_text = article['summary'][:200] + "..." if len(article['summary']) > 200 else article['summary']
                f.write(f"   📄 Tóm tắt: {summary_text}\n\n")
        
        print(f"✅ Đã xuất báo cáo TXT: {filepath}")
        
        # Tự động tải file về (cho Jupyter Notebook)
        self._auto_download_file(filepath, filename, 'text/plain')
        
        return filepath

    def export_to_csv(self, articles: List[Dict], filename: str):
        """Xuất danh sách bài viết ra file CSV và tự động tải về"""
        filepath = os.path.join(Config.OUTPUT_DIR, filename)
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            fieldnames = ['STT', 'Tiêu đề', 'Nguồn', 'Loại nguồn', 'Link', 'Ngày đăng', 'Tóm tắt']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            
            for i, article in enumerate(articles, 1):
                writer.writerow({
                    'STT': i,
                    'Tiêu đề': article['title'],
                    'Nguồn': article['source'],
                    'Loại nguồn': article.get('source_type', 'unknown').upper(),
                    'Link': article['link'],
                    'Ngày đăng': article.get('published', 'N/A'),
                    'Tóm tắt': article['summary'][:300] + '...' if len(article['summary']) > 300 else article['summary']
                })
        
        print(f"✅ Đã xuất danh sách CSV: {filepath}")
        
        # Tự động tải file về (cho Jupyter Notebook)
        self._auto_download_file(filepath, filename, 'text/csv')
        
        return filepath
    
    
    
    def run(self, rss_urls: Optional[List[str]] = None, website_urls: Optional[List[str]] = None) -> Dict:
        """
        Chạy toàn bộ quy trình
        
        Args:
            rss_urls: Danh sách URL RSS feed
            website_urls: Danh sách URL website cần crawl
        
        Returns:
            Dictionary chứa kết quả
        """
        print("\n" + "=" * 80)
        print("🚀 BẮT ĐẦU THU THẬP VÀ TỔNG HỢP TIN TỨC")
        print("=" * 80)
        
        all_articles = []
        
        # 1. Thu thập từ RSS
        if rss_urls:
            rss_articles = self.fetch_rss_feeds(rss_urls)
            all_articles.extend(rss_articles)
        
        # 2. Thu thập từ Website
        if website_urls:
            web_articles = self.fetch_websites(website_urls)
            all_articles.extend(web_articles)
        
        if not all_articles:
            print("\n❌ Không thu thập được bài viết nào!")
            return {}
        
        # 3. Lọc theo chủ đề
        filtered_articles = self.filter_by_topic(all_articles)
        
        if not filtered_articles:
            print("\n❌ Không có bài viết nào phù hợp với chủ đề!")
            return {}
        
        # 4. Tóm tắt
        summary = self.summarize_articles(filtered_articles)
        
        # 5. Đánh giá
        evaluation = self.evaluate_quality(summary, filtered_articles)
        
        # 6. Xuất file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_slug = self.topic.replace(" ", "_")
        txt_file = f"bao_cao_{topic_slug}_{timestamp}.txt"
        csv_file = f"danh_sach_{topic_slug}_{timestamp}.csv"
        
        self.export_to_txt(summary, filtered_articles, evaluation, txt_file)
        self.export_to_csv(filtered_articles, csv_file)
        
        # Kết quả
        print("\n" + "=" * 80)
        print("✅ HOÀN THÀNH!")
        print("=" * 80)
        print(f"\n📊 THỐNG KÊ:")
        print(f"  • Tổng bài viết thu thập: {len(all_articles)}")
        
        if rss_urls:
            rss_count = sum(1 for a in all_articles if a.get('source_type') == 'rss')
            print(f"    - Từ RSS: {rss_count}")
        
        if website_urls:
            web_count = sum(1 for a in all_articles if a.get('source_type') == 'website')
            print(f"    - Từ Website: {web_count}")
        
        print(f"  • Bài viết sau lọc: {len(filtered_articles)}")
        print(f"  • Độ dài tóm tắt: {len(summary)} ký tự")
        print(f"  • Điểm đánh giá: {evaluation.get('overall', 'N/A')}/10")
        print(f"\n📁 FILE OUTPUT:")
        print(f"  • {txt_file}")
        print(f"  • {csv_file}")
        print("\n" + "=" * 80 + "\n")
        
        return {
            'summary': summary,
            'articles': filtered_articles,
            'evaluation': evaluation,
            'stats': {
                'total': len(all_articles),
                'rss': sum(1 for a in all_articles if a.get('source_type') == 'rss'),
                'website': sum(1 for a in all_articles if a.get('source_type') == 'website'),
                'filtered': len(filtered_articles)
            },
            'files': {
                'txt': txt_file,
                'csv': csv_file
            }
        }


