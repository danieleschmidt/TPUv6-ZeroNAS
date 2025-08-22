"""Global-first implementation with internationalization support."""

import logging
from typing import Dict, List, Optional, Any
from enum import Enum

class SupportedLanguage(Enum):
    """Supported languages for global deployment."""
    ENGLISH = "en"
    SPANISH = "es" 
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"

class I18nManager:
    """Internationalization manager for global deployment."""
    
    def __init__(self, default_language: SupportedLanguage = SupportedLanguage.ENGLISH):
        self.default_language = default_language
        self.current_language = default_language
        self.translations = self._load_translations()
        self.logger = logging.getLogger(__name__)
    
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Load translation dictionaries."""
        return {
            "en": {
                "search_started": "Neural architecture search started",
                "search_completed": "Search completed successfully",
                "best_architecture": "Best architecture found",
                "performance_metrics": "Performance metrics",
                "error_occurred": "An error occurred",
                "validation_failed": "Validation failed"
            },
            "es": {
                "search_started": "Búsqueda de arquitectura neuronal iniciada",
                "search_completed": "Búsqueda completada exitosamente", 
                "best_architecture": "Mejor arquitectura encontrada",
                "performance_metrics": "Métricas de rendimiento",
                "error_occurred": "Ocurrió un error",
                "validation_failed": "Validación falló"
            },
            "fr": {
                "search_started": "Recherche d'architecture neuronale commencée",
                "search_completed": "Recherche terminée avec succès",
                "best_architecture": "Meilleure architecture trouvée", 
                "performance_metrics": "Métriques de performance",
                "error_occurred": "Une erreur s'est produite",
                "validation_failed": "Validation échouée"
            },
            "de": {
                "search_started": "Neuronale Architektursuche gestartet",
                "search_completed": "Suche erfolgreich abgeschlossen",
                "best_architecture": "Beste Architektur gefunden",
                "performance_metrics": "Leistungsmetriken", 
                "error_occurred": "Ein Fehler ist aufgetreten",
                "validation_failed": "Validierung fehlgeschlagen"
            },
            "ja": {
                "search_started": "ニューラルアーキテクチャ検索が開始されました",
                "search_completed": "検索が正常に完了しました",
                "best_architecture": "最適なアーキテクチャが見つかりました",
                "performance_metrics": "パフォーマンス指標",
                "error_occurred": "エラーが発生しました", 
                "validation_failed": "検証に失敗しました"
            },
            "zh": {
                "search_started": "神经架构搜索已开始",
                "search_completed": "搜索成功完成",
                "best_architecture": "找到最佳架构",
                "performance_metrics": "性能指标",
                "error_occurred": "发生错误",
                "validation_failed": "验证失败"
            }
        }
    
    def set_language(self, language: SupportedLanguage) -> None:
        """Set current language."""
        self.current_language = language
        self.logger.info(f"Language set to: {language.value}")
    
    def translate(self, key: str, **kwargs) -> str:
        """Translate a key to current language."""
        translations = self.translations.get(self.current_language.value, {})
        translated = translations.get(key, key)
        
        # Format with kwargs if provided
        if kwargs:
            try:
                translated = translated.format(**kwargs)
            except (KeyError, ValueError):
                pass
        
        return translated
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return [lang.value for lang in SupportedLanguage]

# Global instance
i18n = I18nManager()

def t(key: str, **kwargs) -> str:
    """Global translation function."""
    return i18n.translate(key, **kwargs)