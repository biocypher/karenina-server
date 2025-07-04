"""Service for managing rubric state and operations."""

import threading
from typing import Optional

from karenina.schemas.rubric_class import Rubric


class RubricService:
    """Thread-safe service for managing rubric state."""
    
    def __init__(self):
        self._rubric: Optional[Rubric] = None
        self._lock = threading.RLock()
    
    def get_current_rubric(self) -> Optional[Rubric]:
        """Get the current rubric in a thread-safe manner."""
        with self._lock:
            return self._rubric
    
    def set_current_rubric(self, rubric: Optional[Rubric]) -> None:
        """Set the current rubric in a thread-safe manner."""
        with self._lock:
            self._rubric = rubric
    
    def is_rubric_configured(self) -> bool:
        """Check if a rubric is currently configured."""
        with self._lock:
            return self._rubric is not None
    
    def clear_rubric(self) -> None:
        """Clear the current rubric."""
        with self._lock:
            self._rubric = None

    def has_any_rubric(self, finished_templates=None) -> bool:
        """
        Check if there are any rubrics available (global or question-specific).
        
        Args:
            finished_templates: List of finished templates to check for question-specific rubrics
            
        Returns:
            True if global rubric exists OR any template has question_rubric
        """
        # Check global rubric first
        if self.is_rubric_configured():
            return True
        
        # Check for question-specific rubrics
        if finished_templates:
            for template in finished_templates:
                if hasattr(template, 'question_rubric') and template.question_rubric:
                    return True
        
        return False


# Global service instance
rubric_service = RubricService()