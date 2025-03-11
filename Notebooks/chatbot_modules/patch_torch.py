#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Патч для решения проблемы с torch.classes и '__path__._path'.
Этот скрипт должен быть импортирован перед запуском Streamlit.
"""

import sys
import types
import importlib

# Функция для патча модуля torch._classes
def patch_torch_classes():
    try:
        # Проверяем, импортирован ли уже torch
        if 'torch' in sys.modules:
            torch = sys.modules['torch']
            
            # Создаем заглушку для torch._classes, если она еще не существует
            if not hasattr(torch, '_classes'):
                class ClassesStub:
                    def __getattr__(self, name):
                        if name == '__path__':
                            return types.SimpleNamespace(_path=[])
                        return None
                
                torch._classes = ClassesStub()
            
            # Если _classes уже существует, патчим его метод __getattr__
            elif hasattr(torch, '_classes'):
                original_getattr = torch._classes.__getattr__
                
                def patched_getattr(self, name):
                    if name == '__path__':
                        return types.SimpleNamespace(_path=[])
                    return original_getattr(name)
                
                # Заменяем метод __getattr__
                if hasattr(torch._classes, '__getattr__'):
                    torch._classes.__getattr__ = patched_getattr
            
            print("torch._classes успешно пропатчен")
        else:
            print("Модуль torch не импортирован, патч не применен")
    
    except Exception as e:
        print(f"Ошибка при патче torch._classes: {e}")
        import traceback
        print(traceback.format_exc())

# Применяем патч
patch_torch_classes()