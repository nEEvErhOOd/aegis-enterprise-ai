import psutil
import platform
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class SecurityMonitor:
    """Real-time security threat detection with process scanning"""
    
    def __init__(self):
        self.os_type = platform.system()
        self.threat_signatures = self._load_threat_signatures()
        
    def scan_processes(self) -> Dict:
        """Scan running processes for security threats
        
        Returns:
            dict: Threat report with detection status
        """
        threats_detected = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if self._is_threat(proc.info):
                    threats_detected.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return {
            'threats_detected': threats_detected,
            'os_type': self.os_type,
            'status': 'secure' if not threats_detected else 'compromised'
        }
    
    def _is_threat(self, proc_info: Dict) -> bool:
        """Determine if a process matches threat signatures"""
        name = proc_info.get('name', '').lower()
        cmdline = ' '.join(proc_info.get('cmdline', [])).lower()
        
        for signature in self.threat_signatures:
            if signature in name or signature in cmdline:
                return True
        return False
    
    def _load_threat_signatures(self) -> list:
        """Load platform-specific threat signatures"""
        base_signatures = [
            'malware', 'ransomware', 'keylogger', 
            'spyware', 'rootkit', 'trojan'
        ]
        
        if self.os_type == 'Windows':
            return base_signatures + ['powershell -e', 'cmd /c']
        elif self.os_type == 'Darwin':  # macOS
            return base_signatures + ['osascript -e', 'bash -c']
        elif self.os_type == 'Linux':
            return base_signatures + ['sh -c', 'bash -i']
        else:  # Android, iOS
            return base_signatures + [
                'android.runtime', 'ios.system', 
                'jailbreak', 'root'
            ]
