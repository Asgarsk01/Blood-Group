"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useRouter } from "next/navigation"
import { toast } from "sonner"
import { Loader2 } from "lucide-react"
import { saveData, getData, STORAGE_KEYS } from "@/lib/utils/storage"

interface SignupDialogProps {
  children?: React.ReactNode
  variant?: "nav" | "hero"
}

export function SignupDialog({ children, variant = "nav" }: SignupDialogProps) {
  const [fullName, setFullName] = useState("")
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [error, setError] = useState("")
  const [open, setOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const router = useRouter()

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    setIsLoading(true)

    // Basic validation
    if (!fullName || !email || !password || !confirmPassword) {
      setError("Please fill in all fields")
      setIsLoading(false)
      return
    }

    if (!email.endsWith("@gmail.com")) {
      setError("Please use a Gmail address")
      setIsLoading(false)
      return
    }

    if (password !== confirmPassword) {
      setError("Passwords do not match")
      setIsLoading(false)
      return
    }

    if (password.length < 6) {
      setError("Password must be at least 6 characters")
      setIsLoading(false)
      return
    }

    try {
      // Simulate API call to register user
      await new Promise((resolve) => setTimeout(resolve, 1500))

      // Check if user already exists in our "database" (localStorage)
      const users = getData<any[]>(STORAGE_KEYS.REGISTERED_USERS, []);

      if (users.some((user: any) => user.email === email)) {
        setError("An account with this email already exists")
        setIsLoading(false)
        return
      }

      // Add new user to our "database"
      users.push({
        name: fullName,
        email: email,
        password: password,
        createdAt: new Date().toISOString(),
      })

      saveData(STORAGE_KEYS.REGISTERED_USERS, users);

      // Generate a random 6-digit OTP
      const otp = Math.floor(100000 + Math.random() * 900000).toString()

      // Store OTP for verification
      saveData(STORAGE_KEYS.CURRENT_OTP, otp);
      saveData(STORAGE_KEYS.OTP_EMAIL, email);
      saveData(STORAGE_KEYS.OTP_EXPIRY, (Date.now() + 5 * 60 * 1000).toString()); // 5 minutes expiry

      // Store user info for later use
      saveData(STORAGE_KEYS.USER_EMAIL, email);
      saveData(STORAGE_KEYS.USER_NAME, fullName);

      // Send real email with OTP using our API
      const response = await fetch('/api/send-otp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, otp, name: fullName }),
      });

      if (!response.ok) {
        throw new Error('Failed to send verification email');
      }

      // Show toast notification
      toast.success(`A verification code has been sent to ${email}`);

      setOpen(false)
      setIsLoading(false)

      // Navigate to OTP verification
      router.push(`/otp-verification?email=${encodeURIComponent(email)}&name=${encodeURIComponent(fullName)}`)
    } catch (error) {
      console.error("Signup error:", error)
      setError("An error occurred. Please try again.")
      setIsLoading(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild id="signup-trigger">
        {children || (
          <Button
            variant={variant === "nav" ? "outline" : "default"}
            className={variant === "nav" ? "bg-white/10 text-white hover:bg-white/20 hover:text-white" : ""}
          >
            Signup
          </Button>
        )}
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Create an account</DialogTitle>
          <DialogDescription>Sign up to access the Blood Group Detection System.</DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSignup} className="space-y-4 pt-4">
          {error && <div className="bg-destructive/15 text-destructive text-sm p-3 rounded-md">{error}</div>}
          <div className="space-y-2">
            <Label htmlFor="fullName">Full Name</Label>
            <Input
              id="fullName"
              placeholder="John Doe"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              disabled={isLoading}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="email">Gmail Address</Label>
            <Input
              id="email"
              type="email"
              placeholder="your.name@gmail.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              disabled={isLoading}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="password">Password</Label>
            <Input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={isLoading}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="confirmPassword">Confirm Password</Label>
            <Input
              id="confirmPassword"
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              disabled={isLoading}
            />
          </div>
          <Button type="submit" className="w-full" disabled={isLoading}>
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Creating account...
              </>
            ) : (
              "Signup"
            )}
          </Button>
          <div className="text-center text-sm">
            Already have an account?{" "}
            <Button
              variant="link"
              className="p-0 h-auto"
              onClick={() => {
                setOpen(false)
                document.getElementById("login-trigger")?.click()
              }}
              disabled={isLoading}
            >
              Login
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  )
}