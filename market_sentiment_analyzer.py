import requests
from bs4 import BeautifulSoup
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from typing import List, Tuple, Dict, Optional, Any
import logging
import os
import json
from datetime import datetime
from pathlib import Path
import pickle
import hashlib
from PIL import Image
from io import BytesIO
from functools import wraps
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("market_sentiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create cache directories
CACHE_DIR = Path("cache")
IMAGE_CACHE = CACHE_DIR / "images"
DATA_CACHE = CACHE_DIR / "data"
RESULTS_DIR = Path("results")

for directory in [CACHE_DIR, IMAGE_CACHE, DATA_CACHE, RESULTS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

def retry_decorator(max_retries=3, delay=2):
    """
    Decorator for retrying functions that might fail temporarily
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.RequestException, 
                        ConnectionError, 
                        TimeoutError) as e:
                    retries += 1
                    wait_time = delay * (2 ** (retries - 1))  # Exponential backoff
                    logger.warning(
                        f"Retry {retries}/{max_retries} for {func.__name__} after error: {str(e)}. "
                        f"Waiting {wait_time}s"
                    )
                    time.sleep(wait_time)
            
            # If we get here, all retries failed
            logger.error(f"All {max_retries} retries failed for {func.__name__}")
            raise Exception(f"Function {func.__name__} failed after {max_retries} retries")
        return wrapper
    return decorator

class ImageScraper:
    def __init__(self, delay_seconds: int = 2, cache_dir: Path = IMAGE_CACHE):
        self.delay = delay_seconds
        self.headers = {
            'User-Agent': 'Research Bot (educational purposes only)'
        }
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_path(self, url: str) -> Path:
        """Generate a cache path for a URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.jpg"

    @retry_decorator(max_retries=3, delay=2)
    def _download_image(self, url: str) -> Optional[bytes]:
        """Download an image with retries"""
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        return response.content

    def _save_to_cache(self, url: str, image_data: bytes) -> Path:
        """Save image data to cache"""
        cache_path = self._get_cache_path(url)
        with open(cache_path, 'wb') as f:
            f.write(image_data)
        return cache_path

    def _check_cache(self, url: str) -> Optional[Path]:
        """Check if URL exists in cache"""
        cache_path = self._get_cache_path(url)
        if cache_path.exists():
            return cache_path
        return None

    @retry_decorator(max_retries=5, delay=3)
    def _scrape_page(self, url: str, page: int = 1) -> Tuple[List[str], Optional[str]]:
        """
        Scrape a single page of images
        Returns: (list of image URLs, next page URL or None)
        """
        page_url = f"{url}?page={page}" if page > 1 else url
        logger.info(f"Scraping page {page} from {page_url}")
        
        response = requests.get(page_url, headers=self.headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract image URLs - this would need customization for the specific site
        img_tags = soup.find_all('img', class_='post-image')
        image_urls = [img['src'] for img in img_tags if 'src' in img.attrs]
        
        # Look for next page link - this would need customization
        next_page = soup.find('a', class_='next-page')
        next_page_url = next_page['href'] if next_page and 'href' in next_page.attrs else None
        
        return image_urls, next_page_url

    def scrape_images(self, url: str, num_images: int) -> Dict[str, Path]:
        """
        Scrapes image URLs from social media platform with pagination
        Returns: dictionary mapping URLs to local file paths
        """
        logger.info(f"Starting image scraping from {url}, targeting {num_images} images")
        image_urls = []
        page = 1
        next_page_url = url
        
        # Scrape pages until we have enough images or no more pages
        while next_page_url and len(image_urls) < num_images:
            urls, next_page_url = self._scrape_page(next_page_url, page)
            image_urls.extend(urls)
            page += 1
            time.sleep(self.delay)  # Ethical rate limiting
        
        # Limit to the number requested
        image_urls = image_urls[:num_images]
        
        # Download images or use cached versions
        image_paths = {}
        for img_url in image_urls:
            try:
                # Check cache first
                cached_path = self._check_cache(img_url)
                if cached_path:
                    logger.debug(f"Using cached image for {img_url}")
                    image_paths[img_url] = cached_path
                else:
                    # Download and cache if not found
                    logger.debug(f"Downloading image from {img_url}")
                    image_data = self._download_image(img_url)
                    if image_data:
                        cached_path = self._save_to_cache(img_url, image_data)
                        image_paths[img_url] = cached_path
                    time.sleep(self.delay)  # Ethical rate limiting
            except Exception as e:
                logger.error(f"Error processing image {img_url}: {str(e)}")
        
        logger.info(f"Successfully processed {len(image_paths)} images out of {len(image_urls)} URLs")
        return image_paths

class SkirtLengthAnalyzer:
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize base model
        self.model = self._load_model(model_path)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def _load_model(self, model_path: Optional[str]) -> torch.nn.Module:
        """Load model from path or use pretrained model"""
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading custom model from {model_path}")
            model = torch.load(model_path, map_location=self.device)
        else:
            logger.info("Loading pretrained ResNet model for feature extraction")
            # Load pretrained model and modify for our use case
            model = resnet50(pretrained=True)
            
            # Modify the final layer to predict skirt length (0-1)
            num_features = model.fc.in_features
            model.fc = torch.nn.Sequential(
                torch.nn.Linear(num_features, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(256, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1),
                torch.nn.Sigmoid()  # Sigmoid ensures output between 0-1
            )
            
            # In a real scenario, this model would need to be fine-tuned
            # Here we're just demonstrating the architecture
            
        model.to(self.device)
        model.eval()
        return model
    
    def _detect_person_and_skirt(self, image_path: Path) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Use OpenCV to detect a person and potential skirt in the image
        Returns: (success, ROI containing skirt or None)
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                return False, None
                
            # In a real implementation, you would:
            # 1. Use a person detector (HOG, YOLO, or a dedicated model)
            # 2. Run pose estimation to find lower body keypoints
            # 3. Extract the region containing the skirt
            
            # Placeholder implementation (would need to be replaced)
            # This simulates finding a skirt ROI in 70% of images
            if np.random.random() > 0.3:
                # Create a simulated ROI (lower part of the image)
                height, width = image.shape[:2]
                skirt_roi = image[height//2:height, width//4:3*width//4]
                return True, skirt_roi
            else:
                return False, None
                
        except Exception as e:
            logger.error(f"Error in skirt detection for {image_path}: {str(e)}")
            return False, None
    
    def _preprocess_image(self, image_path: Path) -> Optional[torch.Tensor]:
        """Load and preprocess image for model input"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            return image_tensor
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return None

    def analyze_images(self, image_paths: Dict[str, Path]) -> Dict[str, float]:
        """
        Analyzes skirt lengths in images
        Returns: Dictionary mapping URLs to skirt length scores (0-1)
        """
        results = {}
        
        with torch.no_grad():  # Disable gradient calculation for inference
            for url, path in image_paths.items():
                try:
                    # Detect person and skirt
                    has_skirt, skirt_roi = self._detect_person_and_skirt(path)
                    
                    if not has_skirt:
                        logger.debug(f"No skirt detected in {url}")
                        continue
                    
                    # Preprocess image for model
                    image_tensor = self._preprocess_image(path)
                    if image_tensor is None:
                        continue
                    
                    # Run inference
                    prediction = self.model(image_tensor)
                    skirt_length = prediction.item()  # Extract scalar value
                    
                    # Store result
                    results[url] = skirt_length
                    logger.debug(f"Image {url}: skirt length score = {skirt_length:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing image {url}: {str(e)}")
        
        logger.info(f"Successfully analyzed {len(results)} images")
        return results

class ConsumerDataAnalyzer:
    def __init__(self, coffee_data_path: str, haircut_data_path: str):
        self.coffee_data_path = coffee_data_path
        self.haircut_data_path = haircut_data_path
        
        # Load and validate data
        self.coffee_data = self._load_and_validate_data(coffee_data_path, 'spending')
        self.haircut_data = self._load_and_validate_data(haircut_data_path, 'frequency')

    def _load_and_validate_data(self, data_path: str, value_column: str) -> pd.DataFrame:
        """Load, validate, and clean data"""
        try:
            # Load data
            df = pd.read_csv(data_path)
            
            # Check if required column exists
            if value_column not in df.columns:
                raise ValueError(f"Required column '{value_column}' not found in {data_path}")
            
            # Check for missing values
            missing_count = df[value_column].isna().sum()
            if missing_count > 0:
                logger.warning(f"Found {missing_count} missing values in {data_path}")
                df = df.dropna(subset=[value_column])
            
            # Remove invalid values (negative or zero)
            invalid_count = (df[value_column] <= 0).sum()
            if invalid_count > 0:
                logger.warning(f"Found {invalid_count} invalid values in {data_path}")
                df = df[df[value_column] > 0]
            
            logger.info(f"Successfully loaded and cleaned {data_path}: {len(df)} valid records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading or validating data from {data_path}: {str(e)}")
            raise

    def process_consumer_data(self) -> Tuple[float, float]:
        """
        Process coffee spending and haircut frequency data
        Returns normalized values
        """
        # Calculate average monthly coffee spending
        avg_coffee = self.coffee_data['spending'].mean()
        # Calculate average haircut frequency (times per month)
        avg_haircuts = self.haircut_data['frequency'].mean()
        
        # Normalize values between 0 and 1
        norm_coffee = (avg_coffee - self.coffee_data['spending'].min()) / \
                     (self.coffee_data['spending'].max() - self.coffee_data['spending'].min())
        norm_haircuts = (avg_haircuts - self.haircut_data['frequency'].min()) / \
                       (self.haircut_data['frequency'].max() - self.haircut_data['frequency'].min())
        
        logger.info(f"Normalized coffee spending: {norm_coffee:.4f}")
        logger.info(f"Normalized haircut frequency: {norm_haircuts:.4f}")
        
        return norm_coffee, norm_haircuts

class MarketSentimentPredictor:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = self._load_model() if model_path else RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False if model_path is None else True

    def _load_model(self) -> Any:
        """Load trained model from disk"""
        try:
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Loaded model from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from {self.model_path}: {str(e)}")
            return RandomForestRegressor(n_estimators=100, random_state=42)

    def _save_model(self, path: str) -> None:
        """Save model to disk"""
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Saved model to {path}")
        except Exception as e:
            logger.error(f"Error saving model to {path}: {str(e)}")

    def train_model(self, X: np.array, y: np.array) -> Dict[str, float]:
        """
        Train the sentiment prediction model with validation
        """
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training model with {X_train.shape[0]} samples, testing with {X_test.shape[0]} samples")
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        metrics = {
            'rmse': rmse,
            'mae': mae
        }
        
        logger.info(f"Model evaluation: RMSE={rmse:.4f}, MAE={mae:.4f}")
        
        # Save the model
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = model_dir / f"sentiment_model_{timestamp}.pkl"
        self._save_model(str(model_path))
        
        return metrics

    def predict_sentiment(self, features: np.array) -> float:
        """
        Predict market sentiment based on input features
        Returns a value between 0 and 1
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained. Call train_model() first.")
            
        prediction = self.model.predict(features.reshape(1, -1))[0]
        result = np.clip(prediction, 0, 1)
        logger.info(f"Predicted sentiment: {result:.4f}")
        return result

def analyze_market_sentiment(
    social_media_url: str,
    num_images: int,
    coffee_data_path: str,
    haircut_data_path: str,
    model_path: Optional[str] = None,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main function to analyze market sentiment
    Returns dictionary with results and metadata
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Initialize components
        scraper = ImageScraper()
        skirt_analyzer = SkirtLengthAnalyzer(model_path)
        consumer_analyzer = ConsumerDataAnalyzer(coffee_data_path, haircut_data_path)
        predictor = MarketSentimentPredictor(model_path)

        # Gather data
        logger.info("Step 1: Scraping images")
        image_paths = scraper.scrape_images(social_media_url, num_images)
        
        logger.info("Step 2: Analyzing skirt lengths")
        skirt_lengths = skirt_analyzer.analyze_images(image_paths)
        
        logger.info("Step 3: Processing consumer data")
        coffee_score, haircut_score = consumer_analyzer.process_consumer_data()

        # Prepare features
        if skirt_lengths:
            avg_skirt_length = np.mean(list(skirt_lengths.values()))
        else:
            logger.warning("No valid skirt lengths detected, using fallback value")
            avg_skirt_length = 0.5  # Fallback to neutral value
            
        features = np.array([avg_skirt_length, coffee_score, haircut_score])
        
        # Mock training data if needed (in a real scenario, this would be historical data)
        if not predictor.is_trained and model_path is None:
            logger.info("Generating synthetic training data for model training")
            # Generate synthetic data for demonstration
            n_samples = 100
            X_synth = np.random.rand(n_samples, 3)  # 3 features
            # Synthetic formula: longer skirts (lower score), higher coffee spending,
            # and less frequent haircuts correlate with bullish sentiment
            y_synth = (1 - X_synth[:, 0]) * 0.4 + X_synth[:, 1] * 0.3 + (1 - X_synth[:, 2]) * 0.3
            y_synth = np.clip(y_synth + np.random.normal(0, 0.1, n_samples), 0, 1)
            
            logger.info("Step 4: Training model with synthetic data")
            metrics = predictor.train_model(X_synth, y_synth)
        else:
            metrics = {"info": "Using pre-trained model"}
        
        # Predict sentiment
        logger.info("Step 5: Predicting market sentiment")
        sentiment_score = predictor.predict_sentiment(features)
        
        # Prepare results
        results = {
            "timestamp": timestamp,
            "sentiment_score": float(sentiment_score),
            "features": {
                "avg_skirt_length": float(avg_skirt_length),
                "coffee_spending": float(coffee_score),
                "haircut_frequency": float(haircut_score)
            },
            "metadata": {
                "social_media_url": social_media_url,
                "num_images_requested": num_images,
                "num_images_processed": len(image_paths),
                "num_skirts_detected": len(skirt_lengths),
                "processing_time_seconds": time.time() - start_time,
                "model_metrics": metrics
            }
        }
        
        # Save results to file if specified
        if output_file:
            output_path = RESULTS_DIR / output_file
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        
        # Also save to CSV for time series analysis
        csv_path = RESULTS_DIR / "sentiment_history.csv"
        csv_exists = csv_path.exists()
        
        with open(csv_path, 'a') as f:
            if not csv_exists:
                f.write("timestamp,sentiment_score,avg_skirt_length,coffee_spending,haircut_frequency\n")
            f.write(f"{timestamp},{sentiment_score},{avg_skirt_length},{coffee_score},{haircut_score}\n")
        
        logger.info(f"Analysis complete. Sentiment score: {sentiment_score:.4f}")
        return results

    except Exception as e:
        error_msg = f"Error in market sentiment analysis: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "error": error_msg,
            "timestamp": timestamp,
            "success": False
        }

if __name__ == "__main__":
    # Example usage
    results = analyze_market_sentiment(
        social_media_url="https://example.com/skirts",
        num_images=100,
        coffee_data_path="data/coffee_spending.csv",
        haircut_data_path="data/haircut_frequency.csv",
        output_file=f"sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    print(f"\nMarket Sentiment Analysis Results:")
    print(f"Sentiment Score: {results['sentiment_score']:.4f}")
    print(f"Based on:")
    print(f"  - Average Skirt Length: {results['features']['avg_skirt_length']:.4f}")
    print(f"  - Coffee Spending Index: {results['features']['coffee_spending']:.4f}")
    print(f"  - Haircut Frequency Index: {results['features']['haircut_frequency']:.4f}") 