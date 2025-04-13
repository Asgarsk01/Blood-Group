// Storage utility functions to persist data

/**
 * Save data to localStorage, falling back to sessionStorage if necessary
 */
export function saveData(key: string, data: any): void {
  try {
    localStorage.setItem(key, JSON.stringify(data));
  } catch (error) {
    console.warn('Failed to save to localStorage, falling back to sessionStorage:', error);
    try {
      sessionStorage.setItem(key, JSON.stringify(data));
    } catch (innerError) {
      console.error('Failed to save data:', innerError);
    }
  }
}

/**
 * Get data from localStorage, checking sessionStorage as fallback
 */
export function getData<T>(key: string, defaultValue: T): T {
  try {
    // First try localStorage
    const localData = localStorage.getItem(key);
    if (localData) {
      return JSON.parse(localData) as T;
    }

    // Then check sessionStorage (for backward compatibility)
    const sessionData = sessionStorage.getItem(key);
    if (sessionData) {
      // Migrate from session to local storage for future use
      const parsedData = JSON.parse(sessionData);
      saveData(key, parsedData);
      return parsedData as T;
    }
  } catch (error) {
    console.error('Error retrieving data:', error);
  }

  return defaultValue;
}

/**
 * Remove data from both localStorage and sessionStorage
 */
export function removeData(key: string): void {
  try {
    localStorage.removeItem(key);
    sessionStorage.removeItem(key); // Also clear from session for completeness
  } catch (error) {
    console.error('Error removing data:', error);
  }
}

/**
 * Storage keys used in the application for consistency
 */
export const STORAGE_KEYS = {
  REGISTERED_USERS: 'registeredUsers',
  PATIENTS_LIST: 'patientsList',
  SELECTED_PATIENT: 'selectedPatient',
  DETECTION_RESULT: 'detectionResult',
  IS_AUTHENTICATED: 'isAuthenticated',
  USER_EMAIL: 'userEmail',
  USER_NAME: 'userName',
  CURRENT_OTP: 'currentOTP',
  OTP_EMAIL: 'otpEmail',
  OTP_EXPIRY: 'otpExpiry',
}; 