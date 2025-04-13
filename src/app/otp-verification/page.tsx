"use client"

import { useState, useEffect, useRef } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { InputOTP, InputOTPGroup, InputOTPSlot } from "@/components/ui/input-otp"
import { toast } from "sonner"
import { Loader2, Mail, AlertCircle } from "lucide-react"
import { saveData, getData, removeData, STORAGE_KEYS } from "@/lib/utils/storage"

export default function OTPVerification() {
  const [otp, setOtp] = useState("")
  const [countdown, setCountdown] = useState(30)
  const [isResendDisabled, setIsResendDisabled] = useState(true)
  const [isVerifying, setIsVerifying] = useState(false)
  const [isResending, setIsResending] = useState(false)
  const [error, setError] = useState("")
  const [emailSent, setEmailSent] = useState(false)
  const [actualOtp, setActualOtp] = useState<string | null>(null)
  const router = useRouter()
  const searchParams = useSearchParams()
  const email = searchParams.get("email") || ""
  const name = searchParams.get("name") || ""
  const otpInputRef = useRef<HTMLInputElement>(null)

  // Simulate email being sent and get the OTP from storage
  useEffect(() => {
    const simulateEmailSending = async () => {
      await new Promise((resolve) => setTimeout(resolve, 2000))
      setEmailSent(true)

      // Get the OTP from localStorage
      const storedOtp = getData<string>(STORAGE_KEYS.CURRENT_OTP, "");
      const storedEmail = getData<string>(STORAGE_KEYS.OTP_EMAIL, "");

      if (storedOtp && storedEmail === email) {
        setActualOtp(storedOtp)
      } else {
        // Generate a new OTP if none exists
        const newOtp = Math.floor(100000 + Math.random() * 900000).toString()
        saveData(STORAGE_KEYS.CURRENT_OTP, newOtp);
        saveData(STORAGE_KEYS.OTP_EMAIL, email);
        saveData(STORAGE_KEYS.OTP_EXPIRY, (Date.now() + 5 * 60 * 1000).toString());
        setActualOtp(newOtp)
        
        // Send the new OTP via email
        try {
          const response = await fetch('/api/send-otp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
              email, 
              otp: newOtp, 
              name: name || email.split("@")[0].replace(/\./g, " ")
            }),
          });
          
          if (!response.ok) {
            throw new Error('Failed to send verification email');
          }
        } catch (error) {
          console.error("Error sending OTP email:", error);
          // Still continue as the OTP is stored in localStorage
        }
      }
    }

    simulateEmailSending()
  }, [email, name])

  // Countdown timer for resend button
  useEffect(() => {
    let timer: NodeJS.Timeout
    if (countdown > 0 && isResendDisabled) {
      timer = setTimeout(() => setCountdown(countdown - 1), 1000)
    } else {
      setIsResendDisabled(false)
    }
    return () => clearTimeout(timer)
  }, [countdown, isResendDisabled])

  // Check OTP expiry
  useEffect(() => {
    const checkExpiry = () => {
      const expiryTime = getData<string>(STORAGE_KEYS.OTP_EXPIRY, "");
      if (expiryTime && Number.parseInt(expiryTime) < Date.now()) {
        setError("OTP has expired. Please request a new one.")
        setOtp("")
      }
    }

    const interval = setInterval(checkExpiry, 10000) // Check every 10 seconds
    return () => clearInterval(interval)
  }, [])

  const handleResendOTP = async () => {
    setIsResending(true)
    setError("")

    try {
      // Generate a new OTP
      const newOtp = Math.floor(100000 + Math.random() * 900000).toString()
      saveData(STORAGE_KEYS.CURRENT_OTP, newOtp);
      saveData(STORAGE_KEYS.OTP_EXPIRY, (Date.now() + 5 * 60 * 1000).toString());
      setActualOtp(newOtp)
      
      // Send the new OTP via email
      const userName = name || getData<string>(STORAGE_KEYS.USER_NAME, "") || email.split("@")[0].replace(/\./g, " ");
      const response = await fetch('/api/send-otp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          email, 
          otp: newOtp, 
          name: userName 
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to send verification email');
      }

      setCountdown(30)
      setIsResendDisabled(true)
      setIsResending(false)

      toast.success(`A new verification code has been sent to ${email}`);
    } catch (error) {
      console.error("Error resending OTP:", error)
      setError("Failed to resend OTP. Please try again.")
      setIsResending(false)
    }
  }

  const handleVerify = async () => {
    if (otp.length !== 6) {
      setError("Please enter a 6-digit OTP")
      return
    }

    setIsVerifying(true)
    setError("")

    try {
      // Simulate API call to verify OTP
      await new Promise((resolve) => setTimeout(resolve, 1500))

      // Check if OTP is correct
      const storedOtp = getData<string>(STORAGE_KEYS.CURRENT_OTP, "");
      const expiryTime = getData<string>(STORAGE_KEYS.OTP_EXPIRY, "");

      if (!storedOtp || !expiryTime) {
        setError("OTP verification failed. Please request a new OTP.")
        setIsVerifying(false)
        return
      }

      if (Number.parseInt(expiryTime) < Date.now()) {
        setError("OTP has expired. Please request a new one.")
        setIsVerifying(false)
        return
      }

      if (otp !== storedOtp) {
        setError("Invalid OTP. Please check and try again.")
        setIsVerifying(false)
        return
      }

      // OTP is valid, proceed with login

      // Ensure name is stored
      if (name && !getData<string>(STORAGE_KEYS.USER_NAME, "")) {
        saveData(STORAGE_KEYS.USER_NAME, name);
      }

      // Ensure email is stored
      if (email && !getData<string>(STORAGE_KEYS.USER_EMAIL, "")) {
        saveData(STORAGE_KEYS.USER_EMAIL, email);
      }

      // Clear OTP data
      removeData(STORAGE_KEYS.CURRENT_OTP);
      removeData(STORAGE_KEYS.OTP_EXPIRY);

      // Set authentication status
      saveData(STORAGE_KEYS.IS_AUTHENTICATED, "true");

      toast.success("You have been successfully verified");

      // Navigate to dashboard
      router.push("/dashboard")
    } catch (error) {
      console.error("Error verifying OTP:", error)
      setError("Verification failed. Please try again.")
      setIsVerifying(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
      <Card className="w-full max-w-md">
        <CardHeader>
          <div className="flex items-center justify-center mb-4">
            <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center">
              <Mail className="h-8 w-8 text-primary" />
            </div>
          </div>
          <CardTitle className="text-center">OTP Verification</CardTitle>
          <CardDescription className="text-center">
            {emailSent ? (
              <>
                A 6-digit verification code has been sent to <span className="font-medium">{email}</span>
              </>
            ) : (
              <>
                Sending verification code to <span className="font-medium">{email}</span>...
              </>
            )}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {!emailSent ? (
            <div className="flex justify-center py-8">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
          ) : (
            <>
              {error && (
                <div className="bg-destructive/10 text-destructive text-sm p-3 rounded-md flex items-center gap-2">
                  <AlertCircle className="h-4 w-4" />
                  {error}
                </div>
              )}
              <div className="flex justify-center py-4">
                <InputOTP maxLength={6} value={otp} onChange={setOtp} ref={otpInputRef}>
                  <InputOTPGroup>
                    <InputOTPSlot index={0} />
                    <InputOTPSlot index={1} />
                    <InputOTPSlot index={2} />
                    <InputOTPSlot index={3} />
                    <InputOTPSlot index={4} />
                    <InputOTPSlot index={5} />
                  </InputOTPGroup>
                </InputOTP>
              </div>
            </>
          )}
        </CardContent>
        <CardFooter className="flex flex-col space-y-4">
          <Button onClick={handleVerify} className="w-full" disabled={!emailSent || isVerifying || otp.length !== 6}>
            {isVerifying ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Verifying...
              </>
            ) : (
              "Verify & Continue"
            )}
          </Button>
          <div className="text-center text-sm">
            Didn&apos;t receive the code?{" "}
            {isResendDisabled ? (
              <span className="text-muted-foreground">Resend in {countdown}s</span>
            ) : (
              <Button
                variant="link"
                className="p-0 h-auto"
                onClick={handleResendOTP}
                disabled={isResending || !emailSent}
              >
                {isResending ? (
                  <>
                    <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                    Resending...
                  </>
                ) : (
                  "Resend OTP"
                )}
              </Button>
            )}
          </div>
        </CardFooter>
      </Card>
    </div>
  )
}