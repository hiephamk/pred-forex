// import { Box } from "@chakra-ui/react";
// import { GoogleLogin, GoogleOAuthProvider } from "@react-oauth/google";
// import axios from "axios";

// function LoginPage() {
//   const handleLogin = async (credentialResponse: any) => {
//     const token = credentialResponse.credential;
//     if (!token) return;

//     try {
//       const res = await axios.post(
//         "http://localhost:8000/auth/google/login/",
//         { id_token: token }, // send id_token
//         { headers: { "Content-Type": "application/json" } }
//       );

//       const data = res.data;
//       console.log("Django response:", data);

//       if (data.access) {
//         localStorage.setItem("access_token", data.access);
//         localStorage.setItem("refresh_token", data.refresh);
//         window.location.href = "/home";
//       }
//     } catch (err) {
//       console.error("Login failed:", err);
//     }
//   };

//   return (
//     <Box>
//       <GoogleLogin
//         onSuccess={handleLogin}
//         onError={() => console.log("Login Failed")}
//       />
//     </Box>
//   );
// }

// export default function LoginWrapper() {
//   return (
//     <GoogleOAuthProvider clientId="185580666031-1cco9rgkahpue94vjlphi6rjorpfrd1d.apps.googleusercontent.com">
//       <LoginPage />
//     </GoogleOAuthProvider>
//   );
// }

import { Box, Button, Alert, Spinner, VStack } from "@chakra-ui/react";
import { GoogleLogin, GoogleOAuthProvider, CredentialResponse } from "@react-oauth/google";
import axios, { AxiosError } from "axios";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

// Types
interface AuthResponse {
  access: string;
  refresh: string;
  user?: {
    id: string;
    email: string;
    name: string;
  };
}

interface ApiError {
  message?: string;
  detail?: string;
  error?: string;
}

function LoginPage() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>("");
  const navigate = useNavigate();

  const handleLogin = async (credentialResponse: CredentialResponse) => {
    const token = credentialResponse.credential;
    
    if (!token) {
      setError("No credential received from Google");
      return;
    }

    setIsLoading(true);
    setError("");
    const url = import.meta.env.VITE_GOOGLE_LOGIN_PATH
    try {
      const res = await axios.post<AuthResponse>(
        url,
        { id_token: token },
        { 
          headers: { 
            "Content-Type": "application/json" 
          },
          timeout: 10000 // 10 second timeout
        }
      );

      const data = res.data;
      console.log("Login successful:", { user: data.user });

      if (data.access && data.refresh) {
        // Store tokens securely
        localStorage.setItem("access_token", data.access);
        localStorage.setItem("refresh_token", data.refresh);
        
        // Store user info if available
        if (data.user) {
          localStorage.setItem("user_info", JSON.stringify(data.user));
        }

        // Navigate to home page
        navigate("/home", { replace: true });
      } else {
        throw new Error("Invalid response: missing tokens");
      }
    } catch (err) {
      console.error("Login failed:", err);
      
      let errorMessage = "Login failed. Please try again.";
      
      if (axios.isAxiosError(err)) {
        const axiosError = err as AxiosError<ApiError>;
        const status = axiosError.response?.status;
        const data = axiosError.response?.data;
        
        if (status === 400) {
          errorMessage = data?.message || data?.detail || "Invalid Google token";
        } else if (status === 401) {
          errorMessage = "Authentication failed";
        } else if (status === 429) {
          errorMessage = "Too many login attempts. Please try again later.";
        } else if (status && status >= 500) {
          errorMessage = "Server error. Please try again later.";
        } else if (err.code === "ECONNABORTED") {
          errorMessage = "Request timeout. Please check your connection.";
        } else if (!err.response) {
          errorMessage = "Network error. Please check your connection.";
        }
      }
      
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLoginError = () => {
    console.error("Google OAuth error");
    setError("Google authentication failed. Please try again.");
  };

  return (
    <Box maxW="400px" mx="auto" mt={8} p={6} borderWidth={1} borderRadius="lg">
      <VStack gap={4}>
        <Box textAlign="center">
          <h2 style={{ fontSize: "1.5em", fontWeight: "bold", marginBottom: "1em" }}>
            Sign in to your account
          </h2>
        </Box>


        {isLoading ? (
          <Box textAlign="center" py={4}>
            <Spinner size="lg" color="blue.500" />
            <Box mt={2}>Signing in...</Box>
          </Box>
        ) : (
          <GoogleLogin
            onSuccess={handleLogin}
            onError={handleLoginError}
            useOneTap={false}
            theme="outline"
            size="large"
            width="100%"
            text="signin_with"
            shape="rectangular"
          />
        )}

        <Box fontSize="sm" textAlign="center" color="gray.600" mt={4}>
          By signing in, you agree to our Terms of Service and Privacy Policy.
        </Box>
      </VStack>
    </Box>
  );
}

export default function LoginWrapper() {
  const clientId = import.meta.env.VITE_GOOGLE_CLIENT_ID;

  if (!clientId) {
    console.error("Google Client ID is not configured");
    return (
      <Box textAlign="center" mt={8} p={6}>
          Google authentication is not properly configured. Please contact support.
      </Box>
    );
  }

  return (
    <GoogleOAuthProvider clientId={clientId}>
      <LoginPage />
    </GoogleOAuthProvider>
  );
}