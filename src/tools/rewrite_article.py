import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Any
from tqdm.auto import tqdm
import logging

# Add project root to path
project_root = Path(__file__).parent.parent if '__file__' in globals() else Path.cwd().parent
sys.path.insert(0, str(project_root))

from data import load_processed_data
from langchain_ollama import OllamaLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    "ollama_host": "http://127.0.0.1:11434",
    "llm_model": "ministral-3:14b",
    "temperature": 0.7,  # Higher for creative rewriting
    "max_tokens": 1000,
    "batch_size": 1,  # Process one at a time for stability
    "output_file": "train_new.json",
    "checkpoint_interval": 50,  # Save checkpoint every 50 articles
}

# ============================================================================
# Rewriting Prompt Template
# ============================================================================

REWRITE_PROMPT_TEMPLATE = """You are a professional Vietnamese text rewriter for Named Entity Recognition (NER) tasks.

**Task**: Rewrite the following Vietnamese article while preserving ALL named entities EXACTLY as they appear.

**Requirements**:
1. Keep ALL entity mentions (people, organizations, addresses) EXACTLY the same
2. Rewrite the surrounding text naturally in Vietnamese
3. Maintain the same meaning and context
4. Improve readability and fluency
5. Keep the article length similar (±20%)
6. Do NOT translate entities or change entity spellings
7. Preserve entity titles and descriptors (e.g., "NSND", "CEO", "Công ty")

**Named Entities to Preserve**:
- Persons: {persons}
- Organizations: {organizations}
- Addresses: {addresses}

**Original Article**:
{original_text}

**Rewritten Article** (Vietnamese only, no explanations):
"""

# ============================================================================
# LLM Rewriter Class
# ============================================================================

class ArticleRewriter:
    """Rewrite articles using Ollama LLM while preserving entities"""
    
    def __init__(self, 
                 llm_model: str = "ministral-3:14b",
                 ollama_host: str = "http://127.0.0.1:11434",
                 temperature: float = 0.7,
                 max_tokens: int = 1000):
        
        self.llm = OllamaLLM(
            base_url=ollama_host,
            model=llm_model,
            temperature=temperature,
            num_predict=max_tokens,
            top_p=0.9,
            top_k=40
        )
        logger.info(f"✓ Initialized ArticleRewriter with {llm_model}")
    
    def build_prompt(self, article: Dict[str, Any]) -> str:
        """Build rewriting prompt with entity preservation instructions"""
        
        ground_truth = article['ground_truth']
        
        # Format entities for prompt
        persons = ", ".join(ground_truth.get('person', [])) or "None"
        organizations = ", ".join(ground_truth.get('organizations', [])) or "None"
        addresses = ", ".join(ground_truth.get('address', [])) or "None"
        
        prompt = REWRITE_PROMPT_TEMPLATE.format(
            persons=persons,
            organizations=organizations,
            addresses=addresses,
            original_text=article['text']
        )
        print(len(prompt))
        print(prompt)
        
        return prompt
    
    def rewrite_article(self, article: Dict[str, Any]) -> str:
        """Rewrite a single article using LLM"""
        
        try:
            prompt = self.build_prompt(article)
            
            # Call LLM
            start_time = time.time()
            rewritten_text = self.llm.invoke(prompt)
            elapsed = time.time() - start_time
            
            # Clean up response
            rewritten_text = rewritten_text.strip()
            
            logger.debug(f"Rewrote article {article['id']} in {elapsed:.2f}s")
            
            return rewritten_text
        
        except Exception as e:
            logger.error(f"Failed to rewrite article {article['id']}: {e}")
            # Fallback: return original text
            return article['text']
    
    def verify_entities_preserved(self, 
                                   original_entities: Dict[str, List[str]], 
                                   rewritten_text: str) -> Dict[str, Any]:
        """Verify that entities are preserved in rewritten text"""
        
        verification = {
            "all_preserved": True,
            "missing_entities": [],
            "preserved_count": 0,
            "total_count": 0
        }
        
        for entity_type, entities in original_entities.items():
            for entity in entities:
                verification["total_count"] += 1
                if entity in rewritten_text:
                    verification["preserved_count"] += 1
                else:
                    verification["all_preserved"] = False
                    verification["missing_entities"].append({
                        "type": entity_type,
                        "entity": entity
                    })
        
        return verification

# ============================================================================
# Main Rewriting Pipeline
# ============================================================================

def rewrite_training_data(config: Dict[str, Any]):
    """Main function to rewrite all training data"""
    
    logger.info("=" * 80)
    logger.info("TRAINING DATA REWRITER")
    logger.info("=" * 80)
    
    # Step 1: Load training data
    logger.info("\n1. Loading VLSP 2018 NER dataset...")
    data_splits = load_processed_data()
    train_data = data_splits['train']
    
    logger.info(f"✓ Loaded {len(train_data)} training examples")
    
    # Step 2: Initialize rewriter
    logger.info("\n2. Initializing ArticleRewriter...")
    rewriter = ArticleRewriter(
        llm_model=config['llm_model'],
        ollama_host=config['ollama_host'],
        temperature=config['temperature'],
        max_tokens=config['max_tokens']
    )
    
    # Step 3: Rewrite articles
    logger.info("\n3. Rewriting articles...")
    logger.info(f"   Model: {config['llm_model']}")
    logger.info(f"   Temperature: {config['temperature']}")
    logger.info(f"   Total articles: {len(train_data)}")
    
    rewritten_data = []
    stats = {
        "total": len(train_data),
        "success": 0,
        "failed": 0,
        "entities_preserved": 0,
        "entities_missing": 0,
        "total_time": 0
    }
    
    start_time = time.time()
    
    for i, article in enumerate(tqdm(train_data, desc="Rewriting articles")):
        try:
            # Rewrite article
            rewritten_text = rewriter.rewrite_article(article)
            
            # Verify entity preservation
            verification = rewriter.verify_entities_preserved(
                article['ground_truth'],
                rewritten_text
            )
            
            # Update stats
            stats["success"] += 1
            stats["entities_preserved"] += verification["preserved_count"]
            stats["entities_missing"] += len(verification["missing_entities"])
            
            # Warn if entities are missing
            if not verification["all_preserved"]:
                logger.warning(
                    f"Article {article['id']}: {len(verification['missing_entities'])} "
                    f"entities missing in rewrite"
                )
            
            # Create output record
            rewritten_record = {
                "id": article['id'],
                "topic": article.get('topic', 'general'),
                "title": article.get('title', ''),
                "text": article['text'],
                "rewrite": rewritten_text,
                "ground_truth": article['ground_truth']
            }
            
            rewritten_data.append(rewritten_record)
            
            # Save checkpoint periodically
            if (i + 1) % config['checkpoint_interval'] == 0:
                checkpoint_file = f"checkpoint_{i+1}.json"
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(rewritten_data, f, ensure_ascii=False, indent=2)
                logger.info(f"✓ Saved checkpoint: {checkpoint_file}")
        
        except Exception as e:
            logger.error(f"Failed to process article {article['id']}: {e}")
            stats["failed"] += 1
            
            # Add original as fallback
            rewritten_record = {
                "id": article['id'],
                "topic": article.get('topic', 'general'),
                "title": article.get('title', ''),
                "text": article['text'],
                "rewrite": article['text'],  # Fallback to original
                "ground_truth": article['ground_truth']
            }
            rewritten_data.append(rewritten_record)
    
    stats["total_time"] = time.time() - start_time
    
    # Step 4: Save final output
    logger.info("\n4. Saving rewritten data...")
    output_path = Path(config['output_file'])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(rewritten_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ Saved {len(rewritten_data)} rewritten articles to {output_path}")
    
    # Step 5: Print statistics
    logger.info("\n" + "=" * 80)
    logger.info("REWRITING STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total articles:        {stats['total']}")
    logger.info(f"Successfully rewritten: {stats['success']}")
    logger.info(f"Failed:                {stats['failed']}")
    logger.info(f"Total time:            {stats['total_time']:.1f}s")
    logger.info(f"Avg time per article:  {stats['total_time']/stats['total']:.2f}s")
    logger.info(f"\nEntity Preservation:")
    logger.info(f"  Entities preserved:  {stats['entities_preserved']}")
    logger.info(f"  Entities missing:    {stats['entities_missing']}")
    
    total_entities = stats['entities_preserved'] + stats['entities_missing']
    if total_entities > 0:
        preservation_rate = (stats['entities_preserved'] / total_entities) * 100
        logger.info(f"  Preservation rate:   {preservation_rate:.1f}%")
    
    logger.info("=" * 80)
    
    return rewritten_data, stats

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main execution"""
    
    try:
        rewritten_data, stats = rewrite_training_data(CONFIG)
        
        logger.info("\n✅ Rewriting completed successfully!")
        logger.info(f"   Output file: {CONFIG['output_file']}")
        logger.info(f"   Total articles: {len(rewritten_data)}")
        
        # Show sample
        if rewritten_data:
            sample = rewritten_data[0]
            logger.info("\n📄 Sample rewritten article:")
            logger.info(f"   ID: {sample['id']}")
            logger.info(f"   Topic: {sample['topic']}")
            logger.info(f"   Original length: {len(sample['text'])} chars")
            logger.info(f"   Rewritten length: {len(sample['rewrite'])} chars")
            logger.info(f"   Entities: {sum(len(v) for v in sample['ground_truth'].values())} total")
    
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Process interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
