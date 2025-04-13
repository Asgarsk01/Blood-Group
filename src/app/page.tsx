import { Button } from "@/components/ui/button"
import { LoginDialog } from "@/components/login-dialog"
import { SignupDialog } from "@/components/signup-dialog"
import Image from "next/image"
import Link from "next/link"

export default function Home() {
  return (
    <div className="relative min-h-screen flex flex-col">
      {/* Hero Background */}
      <div className="absolute inset-0 z-0">
        <Image
          src="/placeholder.jpg"
          alt="Medical background"
          fill
          className="object-cover brightness-[0.45]"
          priority
        />
      </div>

      {/* Navigation */}
      <header className="relative z-10 w-full py-4 px-6">
        <div className="container mx-auto flex justify-between items-center">
          <h1 className="text-2xl font-bold text-white">Blood Group Detection System</h1>
          <div className="flex gap-4">
            <LoginDialog />
            <SignupDialog />
          </div>
        </div>
      </header>

      {/* Hero Content */}
      <main className="relative z-10 flex-1 flex flex-col items-center justify-center text-center px-6">
        <div className="max-w-3xl mx-auto">
          <h1 className="text-4xl md:text-6xl font-bold text-white mb-4">
            Automated Blood Typing for Enhanced Patient Care
          </h1>
          <p className="text-xl text-white/90 mb-8">
            A deep learning and image processing solution for blood group identification.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <LoginDialog variant="hero">
            <Button size="lg" variant="outline" className="default">
                Get Started
              </Button>
            </LoginDialog>
            <SignupDialog variant="hero">
              <Button size="lg" variant="outline" className="bg-white/10 text-white hover:bg-white/20 hover:text-white">
                Create Account
              </Button>
            </SignupDialog>
          </div>
        </div>
      </main>
    </div>
  )
}