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

interface LoginDialogProps {
  children?: React.ReactNode
  variant?: "nav" | "hero"
}

export function LoginDialog({ children, variant = "nav" }: LoginDialogProps) {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState("")
  const [open, setOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const router = useRouter()

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    setIsLoading(true)

    // Basic validation
    if (!email || !password) {
      setError("Please fill in all fields")
      setIsLoading(false)
      return
    }

    if (!email.endsWith("@gmail.com")) {
      setError("Please use a Gmail address")
      setIsLoading(false)
      return
    }

    try {
      // Simulate API call to authenticate user
      await new Promise((resolve) => setTimeout(resolve, 1500))

      // Check if user exists in our "database" (localStorage)
      const users = getData<any[]>(STORAGE_KEYS.REGISTERED_USERS, []);

      const user = users.find((u: any) => u.email === email)

      if (!user) {
        setError("No account found with this email. Please sign up.")
        setIsLoading(false)
        return
      }

      if (user.password !== password) {
        setError("Incorrect password. Please try again.")
        setIsLoading(false)
        return
      }

      // Generate a random 6-digit OTP
      const otp = Math.floor(100000 + Math.random() * 900000).toString()

      // Store OTP for verification
      saveData(STORAGE_KEYS.CURRENT_OTP, otp);
      saveData(STORAGE_KEYS.OTP_EMAIL, email);
      saveData(STORAGE_KEYS.OTP_EXPIRY, (Date.now() + 5 * 60 * 1000).toString()); // 5 minutes expiry

      // Store user info for later use
      saveData(STORAGE_KEYS.USER_EMAIL, email);

      // Extract name from email or use stored name
      if (user.name) {
        saveData(STORAGE_KEYS.USER_NAME, user.name);
      } else {
        const name = email.split("@")[0].replace(/\./g, " ")
        saveData(STORAGE_KEYS.USER_NAME, name.charAt(0).toUpperCase() + name.slice(1));
      }

      // Send real email with OTP using our API
      const userName = user.name || email.split("@")[0].replace(/\./g, " ");
      const response = await fetch('/api/send-otp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          email, 
          otp, 
          name: userName 
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to send verification email');
      }

      // Show toast notification
      toast.success(`A verification code has been sent to ${email}`);

      setOpen(false)
      setIsLoading(false)

      // Navigate to OTP verification
      router.push(`/otp-verification?email=${encodeURIComponent(email)}`)
    } catch (error) {
      console.error("Login error:", error)
      setError("An error occurred. Please try again.")
      setIsLoading(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild id="login-trigger">
        {children || (
          <Button
            variant={variant === "nav" ? "outline" : "default"}
            className={variant === "nav" ? "bg-white/10 text-white hover:bg-white/20 hover:text-white" : ""}
          >
            Login
          </Button>
        )}
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Login to your account</DialogTitle>
          <DialogDescription>
            Enter your Gmail credentials to access the Blood Group Detection System.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleLogin} className="space-y-4 pt-4">
          {error && <div className="bg-destructive/15 text-destructive text-sm p-3 rounded-md">{error}</div>}
          <div className="space-y-2">
            <Label htmlFor="email">Gmail Address</Label>
            <Input
              id="email"
              type="email"
              placeholder="your.name@gmail.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              disabled={isLoading}
              className="border focus:ring-2 ring-[var(--ring)] ring-offset-0 shadow-none"
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
          <Button type="submit" className="w-full" disabled={isLoading}>
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Logging in...
              </>
            ) : (
              "Login"
            )}
          </Button>
          <div className="text-center text-sm">
            Don&apos;t have an account?{" "}
            <Button
              variant="link"
              className="p-0 h-auto"
              onClick={() => {
                setOpen(false)
                document.getElementById("signup-trigger")?.click()
              }}
              disabled={isLoading}
            >
              Signup
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  )
}