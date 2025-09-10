#!/usr/bin/env python3
"""
Enterprise Architecture Diagram Extractor
Production-grade system for extracting entities, relationships, and metadata 
from complex architecture diagram images using OCR, CV, and NLP.
"""

import re
import cv2
import json
import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from pathlib import Path
from enum import Enum
import concurrent.futures
from functools import lru_cache

# Core dependencies
import pytesseract
import easyocr
from PIL import Image, ImageEnhance, ImageFilter
import networkx as nx
from pydantic import BaseModel, Field, validator
import spacy
from transformers import pipeline

# Computer vision
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComponentType(str, Enum):
    """Comprehensive component type taxonomy for enterprise architecture"""
    # Infrastructure
    SERVER = "Server"
    DATABASE = "Database"
    LOAD_BALANCER = "LoadBalancer"
    FIREWALL = "Firewall"
    ROUTER = "Router"
    SWITCH = "Switch"
    STORAGE = "Storage"
    CACHE = "Cache"
    
    # Applications
    APPLICATION = "Application"
    SERVICE = "Service"
    API_GATEWAY = "APIGateway"
    MESSAGE_QUEUE = "MessageQueue"
    WEB_SERVER = "WebServer"
    APP_SERVER = "ApplicationServer"
    
    # Cloud Services
    CLOUD_FUNCTION = "CloudFunction"
    CONTAINER = "Container"
    KUBERNETES_POD = "KubernetesPod"
    VIRTUAL_MACHINE = "VirtualMachine"
    
    # Network
    SUBNET = "Subnet"
    VPC = "VPC"
    SECURITY_GROUP = "SecurityGroup"
    NAT_GATEWAY = "NATGateway"
    
    # Data
    DATA_LAKE = "DataLake"
    DATA_WAREHOUSE = "DataWarehouse"
    STREAM = "Stream"
    
    # Security
    IDENTITY_PROVIDER = "IdentityProvider"
    CERTIFICATE = "Certificate"
    SECRET_MANAGER = "SecretManager"
    
    # Generic
    COMPONENT = "Component"
    EXTERNAL_SYSTEM = "ExternalSystem"
    USER = "User"
    PROCESS = "Process"

class RelationType(str, Enum):
    """Comprehensive relationship types for enterprise architecture"""
    # Network connections
    HTTP_CALL = "HTTP_CALL"
    HTTPS_CALL = "HTTPS_CALL"
    TCP_CONNECTION = "TCP_CONNECTION"
    UDP_CONNECTION = "UDP_CONNECTION"
    
    # Data flow
    DATA_FLOW = "DATA_FLOW"
    READS_FROM = "READS_FROM"
    WRITES_TO = "WRITES_TO"
    PUBLISHES_TO = "PUBLISHES_TO"
    SUBSCRIBES_TO = "SUBSCRIBES_TO"
    
    # Infrastructure
    HOSTS = "HOSTS"
    DEPLOYED_ON = "DEPLOYED_ON"
    DEPENDS_ON = "DEPENDS_ON"
    USES = "USES"
    
    # Security
    AUTHENTICATES = "AUTHENTICATES"
    AUTHORIZES = "AUTHORIZES"
    ENCRYPTS = "ENCRYPTS"
    
    # Generic
    CONNECTS_TO = "CONNECTS_TO"
    CONTAINS = "CONTAINS"
    INTEGRATES_WITH = "INTEGRATES_WITH"

@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates"""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        return self.width * self.height

@dataclass 
class NetworkInfo:
    """Network-related information"""
    ip_addresses: List[str] = field(default_factory=list)
    ports: List[int] = field(default_factory=list)
    protocols: List[str] = field(default_factory=list)
    hostnames: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)

@dataclass
class SecurityInfo:
    """Security-related information"""
    encryption_methods: List[str] = field(default_factory=list)
    authentication_types: List[str] = field(default_factory=list)
    certificates: List[str] = field(default_factory=list)
    security_groups: List[str] = field(default_factory=list)

class EnterpriseNode(BaseModel):
    """Enhanced node model for enterprise architecture"""
    id: str
    name: str
    type: ComponentType = ComponentType.COMPONENT
    bbox: Optional[BoundingBox] = None
    
    # Network information
    network: NetworkInfo = Field(default_factory=NetworkInfo)
    
    # Security information  
    security: SecurityInfo = Field(default_factory=SecurityInfo)
    
    # Environment and deployment
    environment: Optional[str] = None  # prod, staging, dev
    region: Optional[str] = None
    availability_zone: Optional[str] = None
    
    # Technical specifications
    version: Optional[str] = None
    technology_stack: List[str] = Field(default_factory=list)
    
    # Metadata
    description: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    properties: Dict[str, Any] = Field(default_factory=dict)
    
    # OCR source information
    ocr_text: List[str] = Field(default_factory=list)
    confidence: float = 0.0

class EnterpriseRelationship(BaseModel):
    """Enhanced relationship model for enterprise architecture"""
    id: str
    source_id: str
    target_id: str
    type: RelationType = RelationType.CONNECTS_TO
    
    # Network details
    protocol: Optional[str] = None
    port: Optional[int] = None
    direction: str = "bidirectional"  # inbound, outbound, bidirectional
    
    # Data flow information
    data_format: Optional[str] = None  # JSON, XML, binary
    bandwidth: Optional[str] = None
    latency: Optional[str] = None
    
    # Security
    encryption: Optional[str] = None
    authentication_required: bool = False
    
    # Metadata
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0

class EnterpriseKnowledgeGraph(BaseModel):
    """Complete knowledge graph with enterprise metadata"""
    nodes: List[EnterpriseNode] = Field(default_factory=list)
    relationships: List[EnterpriseRelationship] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX graph for analysis"""
        G = nx.DiGraph()
        
        for node in self.nodes:
            G.add_node(node.id, **node.dict(exclude={'id'}))
            
        for rel in self.relationships:
            G.add_edge(rel.source_id, rel.target_id, **rel.dict(exclude={'source_id', 'target_id'}))
            
        return G

class PatternRecognizer:
    """Advanced pattern recognition for enterprise architecture elements"""
    
    # Comprehensive regex patterns
    IP_PATTERN = re.compile(r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b')
    PORT_PATTERN = re.compile(r':(\d{1,5})\b')
    URL_PATTERN = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+', re.IGNORECASE)
    HOSTNAME_PATTERN = re.compile(r'\b[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*\b')
    
    # Protocol patterns
    PROTOCOL_PATTERN = re.compile(r'\b(https?|ftp|ssh|sftp|jdbc|tcp|udp|grpc|mqtt|amqp|smtp|pop3|imap)\b', re.IGNORECASE)
    
    # Cloud patterns
    AWS_ARN_PATTERN = re.compile(r'arn:aws:[^:]+:[^:]*:[^:]*:[^/]+/[^\s]+')
    AZURE_RESOURCE_PATTERN = re.compile(r'/subscriptions/[^/]+/resourceGroups/[^/]+')
    GCP_RESOURCE_PATTERN = re.compile(r'projects/[^/]+/[^/]+/[^/]+')
    
    # Kubernetes patterns
    K8S_PATTERN = re.compile(r'([a-zA-Z0-9-]+)/([a-zA-Z0-9-]+)(?::(\d+))?')
    DOCKER_IMAGE_PATTERN = re.compile(r'[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+:[a-zA-Z0-9._-]+')
    
    # Database patterns
    DB_CONNECTION_PATTERN = re.compile(r'(jdbc|mongodb|postgresql|mysql|oracle|mssql)://[^\s]+', re.IGNORECASE)
    
    # Security patterns
    SSL_CERT_PATTERN = re.compile(r'([a-zA-Z0-9.-]+\.(?:crt|pem|p12|pfx|key))', re.IGNORECASE)
    
    # Component type classification patterns
    COMPONENT_PATTERNS = {
        ComponentType.DATABASE: [
            re.compile(r'\b(database|db|sql|mysql|postgres|oracle|mongodb|redis|cassandra|dynamodb|rds)\b', re.I),
            re.compile(r'\b(data|storage|repository|warehouse)\b', re.I)
        ],
        ComponentType.LOAD_BALANCER: [
            re.compile(r'\b(load.?balancer|lb|alb|nlb|elb|haproxy|nginx|f5)\b', re.I),
            re.compile(r'\b(balance|distribute|proxy)\b', re.I)
        ],
        ComponentType.API_GATEWAY: [
            re.compile(r'\b(api.?gateway|gateway|kong|zuul|ambassador)\b', re.I),
            re.compile(r'\b(api|rest|graphql|endpoint)\b', re.I)
        ],
        ComponentType.CACHE: [
            re.compile(r'\b(cache|redis|memcached|elasticache|caffeine)\b', re.I)
        ],
        ComponentType.MESSAGE_QUEUE: [
            re.compile(r'\b(queue|kafka|sqs|mq|rabbitmq|activemq|pubsub)\b', re.I),
            re.compile(r'\b(message|event|stream)\b', re.I)
        ],
        ComponentType.CONTAINER: [
            re.compile(r'\b(docker|container|pod|k8s|kubernetes|ecs|fargate)\b', re.I)
        ],
        ComponentType.FIREWALL: [
            re.compile(r'\b(firewall|waf|security.?group|acl|iptables)\b', re.I)
        ],
        ComponentType.WEB_SERVER: [
            re.compile(r'\b(web.?server|apache|nginx|iis|tomcat|jetty)\b', re.I)
        ]
    }
    
    @classmethod
    def extract_network_info(cls, text: str) -> NetworkInfo:
        """Extract network-related information from text"""
        network = NetworkInfo()
        
        # Extract IP addresses
        network.ip_addresses = cls.IP_PATTERN.findall(text)
        
        # Extract ports
        port_matches = cls.PORT_PATTERN.findall(text)
        network.ports = [int(port) for port in port_matches if 1 <= int(port) <= 65535]
        
        # Extract protocols
        network.protocols = cls.PROTOCOL_PATTERN.findall(text)
        
        # Extract URLs
        network.urls = cls.URL_PATTERN.findall(text)
        
        # Extract hostnames (more complex logic needed to avoid false positives)
        potential_hostnames = cls.HOSTNAME_PATTERN.findall(text)
        network.hostnames = [match[0] if isinstance(match, tuple) else match 
                           for match in potential_hostnames 
                           if '.' in (match[0] if isinstance(match, tuple) else match)]
        
        return network
    
    @classmethod
    def extract_security_info(cls, text: str) -> SecurityInfo:
        """Extract security-related information from text"""
        security = SecurityInfo()
        
        # Extract SSL certificates
        security.certificates = cls.SSL_CERT_PATTERN.findall(text)
        
        # Extract encryption methods
        encryption_patterns = re.compile(r'\b(tls|ssl|aes|rsa|sha|md5|https|encrypt)\b', re.I)
        security.encryption_methods = list(set(encryption_patterns.findall(text)))
        
        # Extract authentication types
        auth_patterns = re.compile(r'\b(oauth|saml|ldap|ad|jwt|basic.?auth|api.?key)\b', re.I)
        security.authentication_types = list(set(auth_patterns.findall(text)))
        
        return security
    
    @classmethod
    def classify_component_type(cls, text: str) -> ComponentType:
        """Classify component type based on text content"""
        text_lower = text.lower()
        
        for component_type, patterns in cls.COMPONENT_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(text):
                    return component_type
        
        return ComponentType.COMPONENT
    
    @classmethod
    def infer_relationship_type(cls, source_text: str, target_text: str, 
                              arrow_text: str = "") -> RelationType:
        """Infer relationship type based on component types and context"""
        combined_text = f"{source_text} {target_text} {arrow_text}".lower()
        
        # Database relationships
        if any(db_word in combined_text for db_word in ['database', 'db', 'sql', 'query']):
            if any(read_word in combined_text for read_word in ['read', 'select', 'get']):
                return RelationType.READS_FROM
            elif any(write_word in combined_text for write_word in ['write', 'insert', 'update', 'save']):
                return RelationType.WRITES_TO
        
        # API relationships
        if any(api_word in combined_text for api_word in ['api', 'rest', 'http', 'service']):
            return RelationType.HTTP_CALL if 'https' not in combined_text else RelationType.HTTPS_CALL
        
        # Message queue relationships
        if any(mq_word in combined_text for mq_word in ['queue', 'publish', 'subscribe', 'message']):
            if 'publish' in combined_text:
                return RelationType.PUBLISHES_TO
            elif 'subscribe' in combined_text:
                return RelationType.SUBSCRIBES_TO
        
        # Infrastructure relationships
        if any(infra_word in combined_text for infra_word in ['deploy', 'host', 'run']):
            return RelationType.DEPLOYED_ON
        
        return RelationType.CONNECTS_TO

class ImagePreprocessor:
    """Advanced image preprocessing for optimal OCR results"""
    
    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """Apply comprehensive image enhancement for OCR"""
        # Convert to PIL for better enhancement options
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Enhance contrast and brightness
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.5)
        
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        # Sharpen the image
        pil_image = pil_image.filter(ImageFilter.SHARPEN)
        
        # Convert back to OpenCV format
        enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return enhanced
    
    @staticmethod
    def denoise_image(image: np.ndarray) -> np.ndarray:
        """Remove noise from image"""
        # Use Non-local Means Denoising
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        
        return denoised
    
    @staticmethod
    def detect_text_regions(image: np.ndarray) -> List[BoundingBox]:
        """Detect text regions in the image using EAST text detector or MSER"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Use MSER (Maximally Stable Extremal Regions) for text detection
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        text_regions = []
        for region in regions:
            # Get bounding rectangle for each region
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
            
            # Filter by size (remove very small or very large regions)
            if 10 < w < image.shape[1] * 0.8 and 10 < h < image.shape[0] * 0.8:
                text_regions.append(BoundingBox(x, y, w, h))
        
        return text_regions

class OCREngine:
    """Multi-engine OCR system with fallback and confidence scoring"""
    
    def __init__(self):
        self.tesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.:-/_()[]{},'
        self.easyocr_reader = easyocr.Reader(['en'])
    
    def extract_text_tesseract(self, image: np.ndarray, bbox: Optional[BoundingBox] = None) -> List[Tuple[str, float, BoundingBox]]:
        """Extract text using Tesseract OCR"""
        target_image = image
        if bbox:
            target_image = image[bbox.y:bbox.y+bbox.height, bbox.x:bbox.x+bbox.width]
        
        # Get detailed data from Tesseract
        data = pytesseract.image_to_data(target_image, config=self.tesseract_config, output_type=pytesseract.Output.DICT)
        
        results = []
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if text and int(data['conf'][i]) > 30:  # Confidence threshold
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                
                # Adjust coordinates if we're working with a sub-region
                if bbox:
                    x += bbox.x
                    y += bbox.y
                
                text_bbox = BoundingBox(x, y, w, h, confidence=float(data['conf'][i]))
                results.append((text, float(data['conf'][i]), text_bbox))
        
        return results
    
    def extract_text_easyocr(self, image: np.ndarray, bbox: Optional[BoundingBox] = None) -> List[Tuple[str, float, BoundingBox]]:
        """Extract text using EasyOCR"""
        target_image = image
        if bbox:
            target_image = image[bbox.y:bbox.y+bbox.height, bbox.x:bbox.x+bbox.width]
        
        results = []
        try:
            ocr_results = self.easyocr_reader.readtext(target_image)
            
            for (bbox_coords, text, confidence) in ocr_results:
                if confidence > 0.3:  # Confidence threshold
                    # bbox_coords is a list of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    x_coords = [point[0] for point in bbox_coords]
                    y_coords = [point[1] for point in bbox_coords]
                    
                    x = min(x_coords)
                    y = min(y_coords)
                    w = max(x_coords) - x
                    h = max(y_coords) - y
                    
                    # Adjust coordinates if we're working with a sub-region
                    if bbox:
                        x += bbox.x
                        y += bbox.y
                    
                    text_bbox = BoundingBox(int(x), int(y), int(w), int(h), confidence=confidence)
                    results.append((text, confidence, text_bbox))
        except Exception as e:
            logger.warning(f"EasyOCR failed: {e}")
        
        return results
    
    def extract_text_multi_engine(self, image: np.ndarray, bbox: Optional[BoundingBox] = None) -> List[Tuple[str, float, BoundingBox]]:
        """Extract text using multiple OCR engines and combine results"""
        tesseract_results = self.extract_text_tesseract(image, bbox)
        easyocr_results = self.extract_text_easyocr(image, bbox)
        
        # Combine and deduplicate results
        all_results = tesseract_results + easyocr_results
        
        # Simple deduplication based on text content and proximity
        unique_results = []
        for text, confidence, text_bbox in all_results:
            is_duplicate = False
            for existing_text, existing_conf, existing_bbox in unique_results:
                # Check if texts are similar and bounding boxes overlap significantly
                if (text.lower() == existing_text.lower() and 
                    self._boxes_overlap(text_bbox, existing_bbox, threshold=0.7)):
                    # Keep the result with higher confidence
                    if confidence > existing_conf:
                        unique_results.remove((existing_text, existing_conf, existing_bbox))
                        unique_results.append((text, confidence, text_bbox))
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append((text, confidence, text_bbox))
        
        return unique_results
    
    def _boxes_overlap(self, box1: BoundingBox, box2: BoundingBox, threshold: float = 0.5) -> bool:
        """Check if two bounding boxes overlap significantly"""
        # Calculate intersection
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.width, box2.x + box2.width)
        y2 = min(box1.y + box1.height, box2.y + box2.height)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        intersection = (x2 - x1) * (y2 - y1)
        union = box1.area + box2.area - intersection
        
        iou = intersection / union if union > 0 else 0
        return iou > threshold

class ComponentDetector:
    """Detect architectural components using computer vision and OCR"""
    
    def __init__(self):
        self.ocr_engine = OCREngine()
        self.preprocessor = ImagePreprocessor()
    
    def detect_shapes(self, image: np.ndarray) -> List[BoundingBox]:
        """Detect rectangular shapes that likely represent components"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for contour in contours:
            # Approximate contour to reduce points
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Filter for rectangular shapes (4 points)
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size and aspect ratio
                if (50 < w < image.shape[1] * 0.8 and 
                    30 < h < image.shape[0] * 0.8 and
                    0.2 < w/h < 5.0):  # Reasonable aspect ratio
                    shapes.append(BoundingBox(x, y, w, h))
        
        return shapes
    
    def extract_components(self, image: np.ndarray) -> List[EnterpriseNode]:
        """Extract components from image using shape detection and OCR"""
        # Preprocess image
        enhanced_image = self.preprocessor.enhance_image(image)
        denoised_image = self.preprocessor.denoise_image(enhanced_image)
        
        # Detect shapes
        shapes = self.detect_shapes(denoised_image)
        
        # Also detect text regions
        text_regions = self.preprocessor.detect_text_regions(denoised_image)
        
        # Combine and deduplicate regions
        all_regions = shapes + text_regions
        all_regions = self._merge_overlapping_regions(all_regions)
        
        components = []
        for i, region in enumerate(all_regions):
            # Extract text from region
            text_results = self.ocr_engine.extract_text_multi_engine(denoised_image, region)
            
            if text_results:
                # Combine all text from the region
                all_text = " ".join([text for text, _, _ in text_results])
                avg_confidence = sum([conf for _, conf, _ in text_results]) / len(text_results)
                
                # Extract structured information
                network_info = PatternRecognizer.extract_network_info(all_text)
                security_info = PatternRecognizer.extract_security_info(all_text)
                component_type = PatternRecognizer.classify_component_type(all_text)
                
                # Create node
                node = EnterpriseNode(
                    id=f"component_{i}",
                    name=self._extract_primary_name(all_text),
                    type=component_type,
                    bbox=region,
                    network=network_info,
                    security=security_info,
                    ocr_text=[text for text, _, _ in text_results],
                    confidence=avg_confidence / 100.0  # Normalize to 0-1
                )
                
                components.append(node)
        
        return components
    
    def _merge_overlapping_regions(self, regions: List[BoundingBox], threshold: float = 0.3) -> List[BoundingBox]:
        """Merge overlapping bounding boxes"""
        if not regions:
            return []
        
        # Sort by area (largest first)
        regions.sort(key=lambda r: r.area, reverse=True)
        
        merged = []
        for region in regions:
            overlaps_with_merged = False
            for merged_region in merged:
                if self._boxes_overlap_simple(region, merged_region, threshold):
                    overlaps_with_merged = True
                    break
            
            if not overlaps_with_merged:
                merged.append(region)
        
        return merged
    
    def _boxes_overlap_simple(self, box1: BoundingBox, box2: BoundingBox, threshold: float) -> bool:
        """Simple overlap check"""
        x1_overlap = max(0, min(box1.x + box1.width, box2.x + box2.width) - max(box1.x, box2.x))
        y1_overlap = max(0, min(box1.y + box1.height, box2.y + box2.height) - max(box1.y, box2.y))
        
        overlap_area = x1_overlap * y1_overlap
        min_area = min(box1.area, box2.area)
        
        return (overlap_area / min_area) > threshold if min_area > 0 else False
    
    def _extract_primary_name(self, text: str) -> str:
        """Extract the primary name from OCR text"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return "Unknown"
        
        # Return the longest meaningful line as the primary name
        meaningful_lines = [line for line in lines if len(line) > 2 and not line.isdigit()]
        if meaningful_lines:
            return max(meaningful_lines, key=len)
        
        return lines[0]

class RelationshipDetector:
    """Detect relationships between components using arrow detection and proximity analysis"""
    
    def detect_arrows(self, image: np.ndarray) -> List[Tuple[Tuple[int, int], Tuple[int, int], str]]:
        """Detect arrows in the image and return start/end points with direction"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=30, maxLineGap=10)
        
        arrows = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line properties
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.arctan2(y2-y1, x2-x1)
                
                # Filter for reasonable arrow lengths
                if length > 50:
                    # Detect arrowheads by looking for convergent lines near endpoints
                    direction = "forward" if x2 > x1 else "backward"
                    arrows.append(((x1, y1), (x2, y2), direction))
        
        return arrows
    
    def extract_relationships(self, image: np.ndarray, components: List[EnterpriseNode]) -> List[EnterpriseRelationship]:
        """Extract relationships between components using arrow detection"""
        if len(components) < 2:
            return []
        
        # Detect arrows
        arrows = self.detect_arrows(image)
        
        relationships = []
        relationship_id = 0
        
        # For each arrow, find the closest components
        for start_point, end_point, direction in arrows:
            source_component = self._find_closest_component(start_point, components)
            target_component = self._find_closest_component(end_point, components)
            
            if source_component and target_component and source_component.id != target_component.id:
                # Extract any text along the arrow path
                arrow_text = self._extract_arrow_text(image, start_point, end_point)
                
                # Infer relationship type
                rel_type = PatternRecognizer.infer_relationship_type(
                    " ".join(source_component.ocr_text),
                    " ".join(target_component.ocr_text),
                    arrow_text
                )
                
                # Extract network information from arrow text
                network_info = PatternRecognizer.extract_network_info(arrow_text)
                
                relationship = EnterpriseRelationship(
                    id=f"rel_{relationship_id}",
                    source_id=source_component.id,
                    target_id=target_component.id,
                    type=rel_type,
                    protocol=network_info.protocols[0] if network_info.protocols else None,
                    port=network_info.ports[0] if network_info.ports else None,
                    direction="inbound" if direction == "backward" else "outbound",
                    description=arrow_text if arrow_text else None,
                    confidence=0.7  # Base confidence for arrow-detected relationships
                )
                
                relationships.append(relationship)
                relationship_id += 1
        
        # Add proximity-based relationships for components without arrows
        proximity_relationships = self._detect_proximity_relationships(components)
        relationships.extend(proximity_relationships)
        
        return relationships
    
    def _find_closest_component(self, point: Tuple[int, int], components: List[EnterpriseNode]) -> Optional[EnterpriseNode]:
        """Find the component closest to a given point"""
        if not components:
            return None
        
        min_distance = float('inf')
        closest_component = None
        
        for component in components:
            if component.bbox:
                # Calculate distance to component center
                center = component.bbox.center
                distance = euclidean(point, center)
                
                # Check if point is inside the bounding box (give priority)
                if (component.bbox.x <= point[0] <= component.bbox.x + component.bbox.width and
                    component.bbox.y <= point[1] <= component.bbox.y + component.bbox.height):
                    distance *= 0.1  # Strong preference for components containing the point
                
                if distance < min_distance:
                    min_distance = distance
                    closest_component = component
        
        # Only return if reasonably close (within 200 pixels)
        return closest_component if min_distance < 200 else None
    
    def _extract_arrow_text(self, image: np.ndarray, start_point: Tuple[int, int], 
                           end_point: Tuple[int, int]) -> str:
        """Extract text along the arrow path"""
        x1, y1 = start_point
        x2, y2 = end_point
        
        # Create a region around the arrow line
        margin = 20
        min_x = min(x1, x2) - margin
        max_x = max(x1, x2) + margin
        min_y = min(y1, y2) - margin
        max_y = max(y1, y2) + margin
        
        # Ensure bounds are within image
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(image.shape[1], max_x)
        max_y = min(image.shape[0], max_y)
        
        if max_x > min_x and max_y > min_y:
            arrow_region = image[min_y:max_y, min_x:max_x]
            
            # Use OCR to extract text from this region
            ocr_engine = OCREngine()
            text_results = ocr_engine.extract_text_multi_engine(arrow_region)
            
            # Combine all text found
            return " ".join([text for text, _, _ in text_results])
        
        return ""
    
    def _detect_proximity_relationships(self, components: List[EnterpriseNode]) -> List[EnterpriseRelationship]:
        """Detect relationships based on component proximity"""
        relationships = []
        relationship_id = 1000  # Start with high ID to avoid conflicts
        
        for i, comp1 in enumerate(components):
            for j, comp2 in enumerate(components[i+1:], i+1):
                if comp1.bbox and comp2.bbox:
                    # Calculate distance between component centers
                    distance = euclidean(comp1.bbox.center, comp2.bbox.center)
                    
                    # Create relationship if components are reasonably close
                    if distance < 300:  # Adjustable threshold
                        # Infer relationship type based on component types
                        rel_type = self._infer_proximity_relationship_type(comp1, comp2)
                        
                        relationship = EnterpriseRelationship(
                            id=f"prox_rel_{relationship_id}",
                            source_id=comp1.id,
                            target_id=comp2.id,
                            type=rel_type,
                            direction="bidirectional",
                            confidence=0.4,  # Lower confidence for proximity-based
                            properties={"detection_method": "proximity", "distance": str(int(distance))}
                        )
                        
                        relationships.append(relationship)
                        relationship_id += 1
        
        return relationships
    
    def _infer_proximity_relationship_type(self, comp1: EnterpriseNode, comp2: EnterpriseNode) -> RelationType:
        """Infer relationship type based on component types"""
        # Database relationships
        if comp1.type == ComponentType.DATABASE or comp2.type == ComponentType.DATABASE:
            return RelationType.READS_FROM
        
        # API relationships
        if (comp1.type == ComponentType.API_GATEWAY or comp2.type == ComponentType.API_GATEWAY or
            comp1.type == ComponentType.SERVICE or comp2.type == ComponentType.SERVICE):
            return RelationType.HTTP_CALL
        
        # Load balancer relationships
        if comp1.type == ComponentType.LOAD_BALANCER or comp2.type == ComponentType.LOAD_BALANCER:
            return RelationType.CONNECTS_TO
        
        return RelationType.INTEGRATES_WITH

class EnterpriseArchitectureExtractor:
    """Main extractor class that orchestrates the entire extraction process"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.component_detector = ComponentDetector()
        self.relationship_detector = RelationshipDetector()
        
        # Initialize NLP pipeline for advanced text analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def extract_from_image_path(self, image_path: str) -> EnterpriseKnowledgeGraph:
        """Extract knowledge graph from image file path"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        return self.extract_from_image(image, metadata={"source_path": image_path})
    
    def extract_from_image(self, image: np.ndarray, metadata: Dict[str, Any] = None) -> EnterpriseKnowledgeGraph:
        """Main extraction method"""
        logger.info("Starting enterprise architecture extraction...")
        
        # Extract components
        logger.info("Extracting components...")
        components = self.component_detector.extract_components(image)
        logger.info(f"Found {len(components)} components")
        
        # Extract relationships
        logger.info("Extracting relationships...")
        relationships = self.relationship_detector.extract_relationships(image, components)
        logger.info(f"Found {len(relationships)} relationships")
        
        # Post-process and validate
        components = self._post_process_components(components)
        relationships = self._post_process_relationships(relationships, components)
        
        # Create knowledge graph
        kg = EnterpriseKnowledgeGraph(
            nodes=components,
            relationships=relationships,
            metadata={
                "extraction_timestamp": pd.Timestamp.now().isoformat(),
                "image_dimensions": f"{image.shape[1]}x{image.shape[0]}",
                "total_components": len(components),
                "total_relationships": len(relationships),
                **(metadata or {})
            }
        )
        
        logger.info("Extraction completed successfully")
        return kg
    
    def _post_process_components(self, components: List[EnterpriseNode]) -> List[EnterpriseNode]:
        """Post-process components to improve quality"""
        processed = []
        
        for component in components:
            # Clean up component name
            component.name = self._clean_text(component.name)
            
            # Extract additional metadata using NLP
            if self.nlp and component.ocr_text:
                all_text = " ".join(component.ocr_text)
                doc = self.nlp(all_text)
                
                # Extract entities
                for ent in doc.ents:
                    if ent.label_ == "ORG":
                        component.tags["organization"] = ent.text
                    elif ent.label_ == "PRODUCT":
                        component.tags["product"] = ent.text
                    elif ent.label_ == "CARDINAL":
                        component.tags["version"] = ent.text
            
            # Validate and enhance network information
            component.network = self._validate_network_info(component.network)
            
            # Set environment based on keywords
            if any(env in " ".join(component.ocr_text).lower() for env in ["prod", "production"]):
                component.environment = "production"
            elif any(env in " ".join(component.ocr_text).lower() for env in ["dev", "development"]):
                component.environment = "development"
            elif any(env in " ".join(component.ocr_text).lower() for env in ["staging", "stage"]):
                component.environment = "staging"
            
            processed.append(component)
        
        return processed
    
    def _post_process_relationships(self, relationships: List[EnterpriseRelationship], 
                                  components: List[EnterpriseNode]) -> List[EnterpriseRelationship]:
        """Post-process relationships to improve quality"""
        component_map = {comp.id: comp for comp in components}
        processed = []
        
        for rel in relationships:
            # Validate that source and target components exist
            if rel.source_id not in component_map or rel.target_id not in component_map:
                continue
            
            source_comp = component_map[rel.source_id]
            target_comp = component_map[rel.target_id]
            
            # Enhance relationship based on component types
            rel = self._enhance_relationship(rel, source_comp, target_comp)
            
            processed.append(rel)
        
        # Remove duplicate relationships
        return self._deduplicate_relationships(processed)
    
    def _enhance_relationship(self, rel: EnterpriseRelationship, 
                            source: EnterpriseNode, target: EnterpriseNode) -> EnterpriseRelationship:
        """Enhance relationship with additional metadata"""
        # Set default ports based on relationship type and component types
        if rel.type == RelationType.HTTP_CALL and not rel.port:
            rel.port = 80
        elif rel.type == RelationType.HTTPS_CALL and not rel.port:
            rel.port = 443
        
        # Set encryption for HTTPS
        if rel.type == RelationType.HTTPS_CALL:
            rel.encryption = "TLS"
        
        # Database relationships
        if target.type == ComponentType.DATABASE:
            rel.authentication_required = True
            if "sql" in " ".join(target.ocr_text).lower():
                rel.port = rel.port or 3306  # Default MySQL port
        
        return rel
    
    def _deduplicate_relationships(self, relationships: List[EnterpriseRelationship]) -> List[EnterpriseRelationship]:
        """Remove duplicate relationships"""
        seen = set()
        unique = []
        
        for rel in relationships:
            key = (rel.source_id, rel.target_id, rel.type)
            if key not in seen:
                seen.add(key)
                unique.append(rel)
        
        return unique
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep meaningful ones
        text = re.sub(r'[^\w\s\-\.\:\/]', '', text)
        
        return text
    
    def _validate_network_info(self, network: NetworkInfo) -> NetworkInfo:
        """Validate and clean network information"""
        # Validate IP addresses
        validated_ips = []
        for ip in network.ip_addresses:
            try:
                parts = ip.split('.')
                if len(parts) == 4 and all(0 <= int(part) <= 255 for part in parts):
                    validated_ips.append(ip)
            except ValueError:
                continue
        network.ip_addresses = validated_ips
        
        # Validate ports
        network.ports = [port for port in network.ports if 1 <= port <= 65535]
        
        # Clean protocols
        network.protocols = [proto.lower() for proto in network.protocols]
        
        return network
    
    def save_results(self, kg: EnterpriseKnowledgeGraph, output_path: str, format: str = "json"):
        """Save extraction results in various formats"""
        output_path = Path(output_path)
        
        if format.lower() == "json":
            with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(kg.dict(), f, indent=2, default=str, ensure_ascii=False)
        
        elif format.lower() == "cypher":
            self._save_as_cypher(kg, output_path.with_suffix('.cypher'))
        
        elif format.lower() == "graphml":
            nx_graph = kg.to_networkx()
            nx.write_graphml(nx_graph, output_path.with_suffix('.graphml'))
        
        logger.info(f"Results saved to {output_path}")
    
    def _save_as_cypher(self, kg: EnterpriseKnowledgeGraph, output_path: Path):
        """Save knowledge graph as Cypher statements"""
        with open(output_path, 'w', encoding='utf-8') as f:
            # Create nodes
            f.write("// Create nodes\n")
            for node in kg.nodes:
                props = {
                    'id': node.id,
                    'name': node.name,
                    'type': node.type,
                    'environment': node.environment,
                    **node.tags,
                    **node.properties
                }
                
                # Format properties for Cypher
                prop_string = ', '.join([f"{k}: '{v}'" for k, v in props.items() if v is not None])
                f.write(f"CREATE (n:{node.type} {{{prop_string}}});\n")
            
            f.write("\n// Create relationships\n")
            for rel in kg.relationships:
                props = {
                    'type': rel.type,
                    'protocol': rel.protocol,
                    'port': rel.port,
                    'direction': rel.direction,
                    **rel.properties
                }
                
                prop_string = ', '.join([f"{k}: '{v}'" for k, v in props.items() if v is not None])
                f.write(f"MATCH (a {{id: '{rel.source_id}'}}), (b {{id: '{rel.target_id}'}}) ")
                f.write(f"CREATE (a)-[r:{rel.type} {{{prop_string}}}]->(b);\n")

def main():
    """Example usage of the Enterprise Architecture Extractor"""
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser(description='Extract architecture diagrams to knowledge graphs')
    parser.add_argument('image_path', help='Path to the architecture diagram image')
    parser.add_argument('--output', '-o', default='output', help='Output file prefix')
    parser.add_argument('--format', '-f', choices=['json', 'cypher', 'graphml'], 
                       default='json', help='Output format')
    parser.add_argument('--visualize', '-v', action='store_true', 
                       help='Generate visualization of extracted components')
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = EnterpriseArchitectureExtractor()
    
    try:
        # Extract knowledge graph
        kg = extractor.extract_from_image_path(args.image_path)
        
        # Save results
        extractor.save_results(kg, args.output, args.format)
        
        # Print summary
        print(f"\nExtraction Summary:")
        print(f"Components found: {len(kg.nodes)}")
        print(f"Relationships found: {len(kg.relationships)}")
        print(f"Component types: {', '.join(set(node.type for node in kg.nodes))}")
        
        # Generate visualization if requested
        if args.visualize:
            visualize_extraction_results(kg, args.image_path, f"{args.output}_visualization.png")
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise

def visualize_extraction_results(kg: EnterpriseKnowledgeGraph, original_image_path: str, output_path: str):
    """Visualize the extraction results overlaid on the original image"""
    # Load original image
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        logger.error("Could not load original image for visualization")
        return
    
    # Convert to RGB for matplotlib
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(rgb_image)
    
    # Draw component bounding boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(ComponentType)))
    type_color_map = {comp_type: colors[i] for i, comp_type in enumerate(ComponentType)}
    
    for node in kg.nodes:
        if node.bbox:
            rect = patches.Rectangle(
                (node.bbox.x, node.bbox.y), 
                node.bbox.width, 
                node.bbox.height,
                linewidth=2, 
                edgecolor=type_color_map.get(node.type, 'red'), 
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add component label
            ax.text(node.bbox.x, node.bbox.y - 5, f"{node.name}\n({node.type})", 
                   fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Draw relationships as arrows
    for rel in kg.relationships:
        source_node = next((n for n in kg.nodes if n.id == rel.source_id), None)
        target_node = next((n for n in kg.nodes if n.id == rel.target_id), None)
        
        if source_node and target_node and source_node.bbox and target_node.bbox:
            start = source_node.bbox.center
            end = target_node.bbox.center
            
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='blue', alpha=0.7))
            
            # Add relationship label at midpoint
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y, rel.type, fontsize=6, 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
    
    ax.set_title('Enterprise Architecture Extraction Results', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    main()