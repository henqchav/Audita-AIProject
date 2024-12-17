const SALT = "audita2024salt";
const TEST_USER = {
    username: "testuser",
    passwordHash: "0af51bdcb0a3e0463153bca38e7b0628e4430a61534331973e1d8d1516af1cb5"
};

async function hashPassword(password: string): Promise<string> {
    const encoder = new TextEncoder();
    const data = encoder.encode(password + SALT);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

export async function validateCredentials(username: string, password: string): Promise<boolean> {
    const hashedPassword = await hashPassword(password);
    return username === TEST_USER.username && hashedPassword === TEST_USER.passwordHash;
}

export function setAuthToken(isAuthenticated: boolean) {
    if (typeof window !== 'undefined') {
        if (isAuthenticated) {
            localStorage.setItem('auth', 'true');
        } else {
            localStorage.removeItem('auth');
        }
    }
}

export function isAuthenticated(): boolean {
    if (typeof window !== 'undefined') {
        return localStorage.getItem('auth') === 'true';
    }
    return false;
}
